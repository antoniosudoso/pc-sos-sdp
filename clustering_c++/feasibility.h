#ifndef CLUSTERING_FEASIBILITY_H
#define CLUSTERING_FEASIBILITY_H

#include <armadillo>
#include <list>
#include <map>
#include <set>

int is_feasible(arma::mat &Ws, int k, std::map<int, std::set<int>> &ml_map, std::vector<std::pair<int, int>> &cl_pairs);
bool look_for_feasible_clustering(arma::mat &data, int k, bool verbose,
									int n_start, int n_permutations,
									std::map<int, std::set<int>> &ml_map,
									std::vector<std::pair<int, int>> &local_cl,
									std::vector<std::pair<int, int>> &global_ml,
									std::vector<std::pair<int, int>> &global_cl,
									arma::mat &distances);

#endif
