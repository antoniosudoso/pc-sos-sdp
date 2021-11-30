#include <queue>
#include "feasibility.h"
#include "ilp_model.h"
#include "Kmeans.h"

bool look_for_feasible_clustering(arma::mat &data, int k, bool verbose,
									int n_start, int n_permutations,
									std::map<int, std::set<int>> &ml_map,
									std::vector<std::pair<int, int>> &local_cl,
									std::vector<std::pair<int, int>> &global_ml,
									std::vector<std::pair<int, int>> &global_cl,
									arma::mat &distances){
    Kmeans kmeans(data, k, ml_map, local_cl, global_ml, global_cl, verbose);
	return kmeans.findClustering(n_start, n_permutations, distances);
}

int is_bipartite(int n, std::vector<std::pair<int, int>> &cl_pairs){

	std::vector<int> colors(n, -1);
	std::vector<std::vector<int>> neighbours(n);

	for (auto &pair : cl_pairs){
		neighbours[pair.first].push_back(pair.second);
		neighbours[pair.second].push_back(pair.first);
	}
	
	for (int v = 0; v < n; v++){
		if (colors[v] >= 0)
			continue;

		std::queue<int> to_visit;
		colors[v] = 0;
		to_visit.push(v);

		while (!to_visit.empty()){
			int i = to_visit.front();
			to_visit.pop();

			for (int j : neighbours[i]){
				if (colors[j] == colors[i]){
					return 0;
				} else if (colors[j] < 0) {
					colors[j] = 1 - colors[i];
					to_visit.push(j);
				}
			}
		}
	}
	return 1;
}

int is_ilp_feasible(int n, int k, std::vector<std::pair<int, int>> &cl_pairs){

	int feasible = 1;

	try {
		GRBEnv *env = new GRBEnv();
		ILP_model *model = new ILP_gurobi_model(env, n, k);

		model->add_row_sum_constraints();
		model->add_col_sum_constraints();
		model->add_cannot_link_constraints(cl_pairs);

		model->optimize();
		feasible = !std::isinf(model->get_value());

		delete model;
		delete env;
    } catch (GRBException &e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }

	return feasible;
}

int is_feasible(arma::mat &Ws, int k, std::map<int, std::set<int>> &ml_map,
									std::vector<std::pair<int, int>> &cl_pairs){

	int n = Ws.n_rows;

	if (k == 2)
		return is_bipartite(n, cl_pairs);

	return is_ilp_feasible(n, k, cl_pairs);
}
