#include "ub_heuristics.h"
#include "ilp_model.h"
#include "Kmeans.h"
#include "kmeans_util.h"
#include "util.h"

arma::mat centroids_recovery(arma::mat &Ws_shr, arma::mat &Zopt, int k){

    int n = Ws_shr.n_rows;

    // spectral decomposition of Zopt
    arma::vec eigval;
    arma::mat eigvec;
    eig_sym(eigval, eigvec, Zopt);

    arma::mat U = eigvec.cols(n - k, n - 1);
    arma::mat D = arma::diagmat(eigval);
    D = D(arma::span(n - k, n - 1), arma::span(n - k, n - 1));

    arma::mat new_Zopt = U * D * U.t();
    arma::mat C = new_Zopt * Ws_shr;

    // cluster the rows of C
    std::vector<std::pair<int, int>> ml_init = {};
    std::vector<std::pair<int, int>> cl_init = {};
    Kmeans kmeans_init(C, k, ml_init, cl_init, kmeans_verbose);
    kmeans_init.start(kmeans_max_iter, kmeans_n_start, 1);
    return kmeans_init.getCentroids();
}


arma::mat compute_centroids(arma::mat &Ws, arma::sp_mat &assignment_X,
													arma::vec &cardinalities){
	arma::mat centroids = assignment_X.t() * Ws;
	arma::vec clusters_cardinalities = assignment_X.t() * cardinalities;
	centroids = centroids.each_col() / clusters_cardinalities;

	return centroids;
}

arma::sp_mat build_unshrinking_matrix(int n, std::map<int, std::set<int>> &ml_map){
	int n_shr = ml_map.size();
	arma::sp_mat U(n_shr, n);
	for (int i = 0; i < n_shr; i++){
		for (int j : ml_map[i]){
			U(i, j) = 1;
		}
	}
	return U;
}

double ilp_heuristic(arma::mat &Z, arma::mat &Ws, int k, int original_n,
					double C_trace, std::map<int, std::set<int>> &ml_map,
					std::vector<std::pair<int, int>> &local_cl_pairs, double best_ub,
					arma::sp_mat &node_assignment_X, arma::mat &node_centroids) {

	arma::mat init_centroids = centroids_recovery(Ws, Z, k);
	return ilp_heuristic(Z, Ws, k, original_n, C_trace, init_centroids,
											ml_map, local_cl_pairs, best_ub,
											node_assignment_X, node_centroids);
}

double ilp_heuristic(arma::mat &Z, arma::mat &Ws, int k, int original_n,
					double C_trace, arma::mat &init_centroids,
					std::map<int, std::set<int>> &ml_map,
					std::vector<std::pair<int, int>> &local_cl_pairs, double best_ub,
					arma::sp_mat &node_assignment_X, arma::mat &node_centroids) {

	int n = Z.n_rows;
	double loss;
	arma::vec cardinalities(n);
	for (int i = 0; i < n; i++){
		cardinalities(i) = ml_map[i].size();
	}

	try {
		GRBEnv *env = new GRBEnv();
		ILP_model *model = new ILP_gurobi_model(env, n, k);

		model->compute_objective_function_constant(Ws, cardinalities, C_trace);
		model->add_row_sum_constraints();
		model->add_col_sum_constraints();
		model->add_cannot_link_constraints(local_cl_pairs);

		double previous_loss = std::numeric_limits<double>::infinity();
		arma::mat current_centroids = init_centroids;
		arma::sp_mat current_assignments;

		int n_iter = 0;
		while (1){

			model->set_objective_function(Ws, cardinalities, current_centroids);
			model->optimize();
			n_iter++;

			loss = model->get_value();
			if (std::abs(previous_loss - loss) < 1e-06)
				break;
			previous_loss = loss;

			current_assignments = model->get_solution();
			current_centroids = compute_centroids(Ws, current_assignments, cardinalities);
		}
		if (loss < best_ub){
			arma::sp_mat U = build_unshrinking_matrix(original_n, ml_map);
			node_assignment_X = U.t() * model->get_solution();
			node_centroids = current_centroids;
		}

		delete model;
		delete env;
    } catch (GRBException &e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }

	return loss;
}

// our sdp-based heuristic
double cluster_recovery(arma::mat &Ws, arma::mat &Ws_shr, arma::mat &Xopt, int k,
                        std::map<int, std::set<int>> &ml_map, std::vector<std::pair<int, int>> &local_cl,
                        std::vector<std::pair<int, int>> &global_ml, std::vector<std::pair<int, int>> &global_cl,
                        arma::sp_mat &assignment_X, arma::mat &out_centroids) {
    arma::mat init_centroid = centroids_recovery(Ws_shr, Xopt, k);

    // now perform constrained k-means with the smart initialization
    Kmeans kmeans(Ws, k, ml_map, local_cl, global_ml, global_cl, kmeans_verbose);
    bool flag_partition = kmeans.start(kmeans_max_iter, kmeans_permutations, init_centroid);
    if (!flag_partition)
        return std::numeric_limits<double>::infinity();
    assignment_X = kmeans.getAssignments();
    out_centroids = kmeans.getCentroids();
    return kmeans.getLoss();
}

void print_constraints(std::vector<std::pair<int, int>> &constraints){
	for (auto &pair : constraints){
		std::cout << pair.first << " " << pair.second << "\n";
	}
}

void print_clustering(arma::sp_mat &assignment_X){
	int n = assignment_X.n_rows;
	int k = assignment_X.n_cols;
	for (int h = 0; h < k; h++){
		std::cout << "\n";
		for (int i = 0; i < n; i++){
			if (assignment_X(i, h) > 0)
				std::cout << i << " ";
		}
		std::cout << "\n";
	}
}

void get_local_additional_constraints(arma::mat &Z, int k,
							std::vector<std::pair<int, int>> &local_cl_pairs,
							double constant_a, double constant_b,
							std::vector<std::pair<int, int>> &new_ml_pairs,
							std::vector<std::pair<int, int>> &new_cl_pairs){
	int n = Z.n_rows;
	std::vector<std::vector<int>> clusters;
	std::vector<int> colors(n, -1);
	int n_clusters = 0;

	for (int i = 0; i < n; i++){
		if (colors[i] >= 0)
			continue;
		std::vector<int> cluster = {i};
		colors[i] = n_clusters;
		for (int j = i+1; j < n; j++){
			if (Z(i,j) <= constant_a * Z(i,i)){
				new_cl_pairs.push_back(std::pair<int,int>(i, j));
			}
			if (colors[j] <=  0){
				if (Z(i,j) >= constant_b * Z(i,i)){
					cluster.push_back(j);
					colors[j] = n_clusters;
				}
			}
		}
		clusters.push_back(cluster);
		n_clusters++;
	}
	for (auto &pair : local_cl_pairs){
		if (colors[pair.first] == colors[pair.second]){
			int cluster_index = colors[pair.first];
		
			for (auto it = clusters[cluster_index].begin(); it != clusters[cluster_index].end();){
				if (*it == pair.first || *it == pair.second)
					it = clusters[cluster_index].erase(it);
				else
					it++;
			}
			colors[pair.first] = n_clusters++;
			colors[pair.second] = n_clusters++;
			std::vector<int> cluster_first = {pair.first};
			std::vector<int> cluster_second = {pair.second};
			clusters.push_back(cluster_first);
			clusters.push_back(cluster_second);
		}
	}
	for (auto &cluster : clusters){
		int cardinality = cluster.size();
		for (int j = 1; j < cardinality; j++){
			new_ml_pairs.push_back(std::pair<int,int>(cluster[0], cluster[j]));
		}
	}
}

void get_additional_constraints(arma::mat &Z, int k,
							std::map<int, std::set<int>> &ml_map,
							std::vector<std::pair<int, int>> &local_cl_pairs,
							double constant_a, double constant_b,
							std::vector<std::pair<int, int>> &new_ml_pairs,
							std::vector<std::pair<int, int>> &new_cl_pairs){
	int n = Z.n_rows;
	std::vector<std::vector<int>> clusters;
	std::vector<int> colors(n, -1);
	int n_clusters = 0;

	for (int i = 0; i < n; i++){
		if (colors[i] >= 0)
			continue;
		std::vector<int> cluster = {i};
		colors[i] = n_clusters;
		for (int j = i+1; j < n; j++){
			if (Z(i,j) <= constant_a * Z(i,i)){
				int global_i = *ml_map[i].begin();
				int global_j = *ml_map[j].begin();
				new_cl_pairs.push_back(std::pair<int,int>(global_i, global_j));
			}
			if (colors[j] <=  0){
				if (Z(i,j) >= constant_b * Z(i,i)){
					cluster.push_back(j);
					colors[j] = n_clusters;
				}
			}
		}
		clusters.push_back(cluster);
		n_clusters++;
	}
	for (auto &pair : local_cl_pairs){
		if (colors[pair.first] == colors[pair.second]){
			int cluster_index = colors[pair.first];
		
			for (auto it = clusters[cluster_index].begin(); it != clusters[cluster_index].end();){
				if (*it == pair.first || *it == pair.second)
					it = clusters[cluster_index].erase(it);
				else
					it++;
			}
			colors[pair.first] = n_clusters++;
			colors[pair.second] = n_clusters++;
			std::vector<int> cluster_first = {pair.first};
			std::vector<int> cluster_second = {pair.second};
			clusters.push_back(cluster_first);
			clusters.push_back(cluster_second);
		}
	}
	for (auto &cluster : clusters){
		int cardinality = cluster.size();
		int global_i = *ml_map[cluster[0]].begin();
		for (int j = 1; j < cardinality; j++){
			int global_j = *ml_map[cluster[j]].begin();
			new_ml_pairs.push_back(std::pair<int,int>(global_i, global_j));
		}
	}
}

double get_clustering_value(arma::mat &Z, arma::mat &Ws, int k, double C_trace,
							int original_n, std::map<int, std::set<int>> &ml_map,
							std::vector<std::pair<int, int>> &local_cl_pairs,
							arma::sp_mat &assignment_X, arma::mat &centroids){

	int n = Z.n_rows;
	arma::mat Y = arma::zeros(n, n);
	std::vector<std::vector<int>> clusters;
	
	for (int i = 0; i < n; i++){
		if (Y(i,i) > 0)
			continue;
		std::vector<int> cluster = {i};
		int cluster_size = ml_map[i].size();
		for (int j = i+1; j < n; j++){
			if (Y(j,j) > 0)
				continue;
			if (Z(i,j) > Z(i,i) - Z(i,j)){
				cluster.push_back(j);
				cluster_size += ml_map[j].size();
			}
		}
		clusters.push_back(cluster);
		double elem = 1.0/cluster_size;
		for (int h : cluster){
			for (int l : cluster)
				Y(h,l) = elem;
		}
	}
	if (clusters.size() != k){
		return std::numeric_limits<double>::infinity();
	}
	for (auto &pair : local_cl_pairs){
		if (Y(pair.first, pair.second) > 0)
			return std::numeric_limits<double>::infinity();
	}
	
	arma::mat C = Ws * Ws.t();
	C = -C;
	double ret = C_trace + arma::trace(C*Y);

	arma::sp_mat local_assignment_X(n, k);
	for (int j = 0; j < k; j++){
		for (int i : clusters[j]){
			local_assignment_X(i, j) = 1;
		}
	}
	arma::sp_mat U = build_unshrinking_matrix(original_n, ml_map);
	assignment_X = U.t() * local_assignment_X;
	arma::vec cardinalities(n);
	for (int i = 0; i < n; i++){
		cardinalities(i) = ml_map[i].size();
	}
	centroids = compute_centroids(Ws, local_assignment_X, cardinalities);

	return ret;
}

double get_feasible_clustering(arma::mat &Z, arma::mat &Ws, arma::mat &node_Ws, int k, double C_trace,
								std::vector<std::pair<int, int>> &global_ml_pairs,
								std::vector<std::pair<int, int>> &global_cl_pairs,
								std::map<int, std::set<int>> &ml_map,
								std::vector<std::pair<int, int>> &local_cl_pairs,
								arma::sp_mat &node_assignment_X, arma::mat &node_centroids){

	arma::sp_mat assignment_X;
	arma::mat centroids;
	double best_ub;

	std::vector<std::pair<int,int>> new_ml_pairs, new_cl_pairs;
	new_cl_pairs = global_cl_pairs;
	new_ml_pairs = global_ml_pairs;
	get_additional_constraints(Z, k, ml_map, local_cl_pairs,
									-1, 0.75, new_ml_pairs, new_cl_pairs);
	double new_ub_75 = cluster_recovery(Ws, node_Ws, Z, k, ml_map, local_cl_pairs,
								   new_ml_pairs, new_cl_pairs, node_assignment_X, node_centroids);
	best_ub = new_ub_75;

	return best_ub;
}
