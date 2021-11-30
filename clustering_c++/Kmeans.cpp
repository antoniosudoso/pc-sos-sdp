#include "Kmeans.h"

Kmeans::Kmeans(const arma::mat &data, int k, std::map<int, std::set<int>> &ml_map,
					std::vector<std::pair<int, int>> &local_cl_pair,
					std::vector<std::pair<int, int>> &global_ml,
					std::vector<std::pair<int, int>> &global_cl, bool verbose){
    this->k = k;
    this->n = data.n_rows;
    this->d = data.n_cols;
    this->data = data;
    this->verbose = verbose;
    this->loss = std::numeric_limits<double>::infinity();
    this->constraint = transitive_closure(global_ml, global_cl, n);
	this->cl_degrees = get_points_degrees(n, ml_map, local_cl_pair);
	//this->points_sorting = get_points_sorting(this->cl_degrees);

    if (verbose) {
        std::cout << "ML_TRANSITIVE" << "\n";
        display_graph(this->constraint.ml_graph);
        std::cout << "CL_TRANSITIVE" << "\n";
        display_graph(this->constraint.cl_graph);
    }
}

Kmeans::Kmeans(const arma::mat &data, int k,
					std::vector<std::pair<int, int>> &global_ml,
					std::vector<std::pair<int, int>> &global_cl, bool verbose){
    this->k = k;
    this->n = data.n_rows;
    this->d = data.n_cols;
    this->data = data;
    this->verbose = verbose;
    this->loss = std::numeric_limits<double>::infinity();
    this->constraint = transitive_closure(global_ml, global_cl, n);
	this->cl_degrees = std::vector<int>(n, 0);

    if (verbose) {
        std::cout << "ML_TRANSITIVE" << "\n";
        display_graph(this->constraint.ml_graph);
        std::cout << "CL_TRANSITIVE" << "\n";
        display_graph(this->constraint.cl_graph);
    }
}

// mss objective function
double Kmeans::objectiveFunction() {
    arma::sp_mat assignments_mat = getAssignments();
    arma::mat m = data - assignments_mat * centroids;
    return arma::dot(m.as_col(), m.as_col());
}


/*
// pick centroids as random points of the dataset
void Kmeans::initCentroids() {

    //arma::uvec idx = arma::randperm(n, k);
    //centroids = data.rows(idx);

    arma::vec chances = arma::ones(n);
    centroids = arma::zeros(k, d);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int j = 0; j < k; j++) {

        chances = chances / arma::sum(chances);
        double r = dis(gen);
        double acc = 0.0;
        int index;
        for (index = 0; index < n; index++) {
            double chance = chances(index);
            if (acc + chance >= r)
                break;
            acc += chance;
        }
        centroids.row(j) = data.row(index);


        for (int i = 0; i < n; i++) {
            arma::mat sub_centroids = centroids.rows(0, j);
            arma::vec point = data.row(i).t();
            arma::vec distances = closest_clusters(sub_centroids, point);
            int min_i = distances.index_min();
            chances(i) = distances(min_i);

        }
    }
}
*/



// pick centroids as random points of the dataset
void Kmeans::initCentroids(arma::mat &distances) {

    //arma::uvec idx = arma::randperm(n, k);
    //centroids = data.rows(idx);

    arma::vec chances = arma::ones(n);
    centroids = arma::zeros(k, d);
	std::vector<int> centroids_indices;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (int j = 0; j < k; j++) {

        chances = chances / arma::sum(chances);
        double r = dis(gen);
        double acc = 0.0;
        int index;
        for (index = 0; index < n; index++) {
            double chance = chances(index);
            if (acc + chance >= r)
                break;
            acc += chance;
        }
        centroids.row(j) = data.row(index);
		centroids_indices.push_back(index);


        for (int i = 0; i < n; i++) {
            arma::vec dist_i = closest_clusters(centroids_indices, i, distances);
            int min_i = dist_i.index_min();
            chances(i) = dist_i(min_i);
        }
    }
}

// returns false if the constraints are not satisfied
bool Kmeans::assignPoints(std::vector<int> &permutation) {

    // init assignments for each point to -1
    assignments = arma::zeros(n) - 1;

    for (int h = 0; h < n; h++) {

		int i = permutation[h];

		if (assignments(i) > -1)
			continue;

        arma::vec point_i = data.row(i).t();
		// center of gravity considering all must-link constraints
		for (auto &j : constraint.ml_graph[i]) {
			point_i = point_i + data.row(j).t();
		}
		point_i = point_i / (constraint.ml_graph[i].size() + 1);
		
        // point distances from each cluster (cluster_id, distance_point_i)
        std::vector<std::pair<int, double>> map(k);
        for (int j = 0; j < k; j++) {
            arma::vec centroid_j = centroids.row(j).t();
            double dist_j = squared_distance(point_i, centroid_j);
            map[j] = std::pair<int, double>(j, dist_j);
        }
        sort(map.begin(), map.end(), sort_by_value);
        bool found_cluster = false;
        for (auto &elem_map : map) {
            // cluster id
            int index = elem_map.first;
            bool violate = violateConstraint(i, index);
            if (!violate) {
                found_cluster = true;
                assignments(i) = index;
                for (auto &j : constraint.ml_graph[i]) {
                    assignments(j) = index;
                }
                break;
            }
        }
        if (!found_cluster) {
            if (verbose)
                std::fprintf(stderr, "assignPoints(): empty partition!\n");
            return false;
        }
    }
    return true;
}

bool Kmeans::violateConstraint(int point_i, int cluster_j) {

    for (auto &i : constraint.ml_graph[point_i]) {
        int cluster_id = assignments(i);
        if (cluster_id != -1 && cluster_id != cluster_j) {
            return true;
        }
    }

    for (auto &i : constraint.cl_graph[point_i]) {
        int cluster_id = assignments(i);
        if (cluster_id == cluster_j) {
            return true;
        }
    }
    return false;
}

// returns true if an empty cluster is found
bool Kmeans::computeCentroids() {

    centroids = arma::zeros(k, d);
    // count the number of point for each cluster
    arma::vec count = arma::zeros(k);

    // sum up and counts points for each cluster
    for (int i = 0; i < n; i++) {
        int cluster_id = assignments(i);
        centroids.row(cluster_id) += data.row(i);
        count(cluster_id) = count(cluster_id) + 1;
    }

    // divide by count to get new centroids
    for (int j = 0; j < k; j++) {
        // empty cluster
        if (count(j) == 0) {
            if (verbose)
                std::printf("computeCentroids(): cluster %d is empty!\n", j);
            return false;
        }
        centroids.row(j) = centroids.row(j) / count(j);
    }
    return true;
}

bool Kmeans::start(int max_iter, int n_permutations, arma::mat &init_centroids) {

    double best_loss = std::numeric_limits<double>::infinity();
    arma::vec best_assignments;
    arma::mat best_centroids;

    std::mt19937 gen(int(init_centroids(0, 0)));
	for (int j = 0; j < n_permutations; j++){

		std::vector<int> permutation = get_weighted_permutation(this->cl_degrees, gen);

		centroids = init_centroids;
		if (!assignPoints(permutation))
			continue;

		loss = objectiveFunction();

		if (verbose)
			std::printf("iter | loss\n");

		int n_iter = 0;
		for (int h = 0; h < max_iter; h++) {
			n_iter = n_iter + 1;
			if (verbose)
				std::printf("%d | %f\n", n_iter, loss);
			if (!computeCentroids())
				break;
			if (!assignPoints(permutation)){
				loss = std::numeric_limits<double>::infinity();
				break;
			}
			double current_loss = objectiveFunction();
			double diff = std::abs(loss - current_loss);
			loss = current_loss;
			if (diff <= 1e-6) {
				if (verbose)
					std::printf("Stop!\n");
				break;
			}
		}

		// update with best loss
		if (loss < best_loss) {
			best_loss = loss;
			best_assignments = assignments;
			best_centroids = centroids;
		}
	}

    if (std::isinf(best_loss))
        return false;

    loss = best_loss;
    assignments = best_assignments;
    centroids = best_centroids;
    return true;
}


// multi-start
bool Kmeans::start(int max_iter, int n_start, int n_permutations, arma::mat &distances) {

    double best_loss = std::numeric_limits<double>::infinity();
    arma::vec best_assignments;
    arma::mat best_centroids;

    for (int i = 0; i < n_start; i++) {

		initCentroids(distances);
		arma::mat initial_centroids = this->centroids;

		std::mt19937 gen(1);
		for (int j = 0; j < n_permutations; j++){

			std::vector<int> permutation = get_weighted_permutation(this->cl_degrees, gen);

			this->centroids = initial_centroids;
			if (!assignPoints(permutation))
				continue;

			loss = objectiveFunction();

			if (verbose) std::printf("iter | loss\n");

			int n_iter = 0;
			for (int h = 0; h < max_iter; h++) {
				n_iter = n_iter + 1;
				if (verbose)
					std::printf("%d | %f\n", n_iter, loss);
				if (!computeCentroids())
					break;
				if (!assignPoints(permutation)){
					loss = std::numeric_limits<double>::infinity();
					break;
				}
				double current_loss = objectiveFunction();
				double diff = std::abs(loss - current_loss);
				loss = current_loss;
				if (diff <= 1e-6) {
					if (verbose)
						std::printf("Stop!\n");
					break;
				}
			}

			// update with best loss
			if (loss < best_loss) {
				best_loss = loss;
				best_assignments = assignments;
				best_centroids = centroids;
			}
		}
    }

    if (std::isinf(best_loss))
        return false;

    loss = best_loss;
    assignments = best_assignments;
    centroids = best_centroids;
    return true;
}

// multi-start
bool Kmeans::start(int max_iter, int n_start, int n_permutations) {
	arma::mat distances = compute_distances(data);
	return this->start(max_iter, n_start, n_permutations, distances);
}

bool Kmeans::findClustering(int n_start, int n_permutations, arma::mat distances) {

    for (int i = 0; i < n_start; i++) {

		initCentroids(distances);
		arma::mat initial_centroids = this->centroids;

		std::mt19937 gen(1);
		for (int j = 0; j < n_permutations; j++){

			std::vector<int> permutation = get_weighted_permutation(this->cl_degrees, gen);
			if (!assignPoints(permutation))
				continue;
			else
				return true;
		}
    }
	return false;
}

double Kmeans::getLoss() {
    return loss;
}

// get assignments matrix X
arma::sp_mat Kmeans::getAssignments() {
    arma::sp_mat assignments_mat(n, k);
    for (int i = 0; i < n; i++) {
        int cluster_j = assignments(i);
        assignments_mat(i, cluster_j) = 1;
    }
    return assignments_mat;
}

arma::mat Kmeans::getCentroids() {
    return centroids;
}

