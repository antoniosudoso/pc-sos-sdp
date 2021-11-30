#ifndef CLUSTERING_UB_HEURISTICS_H
#define CLUSTERING_UB_HEURISTICS_H

#include <armadillo>
#include <set>

double ilp_heuristic(arma::mat &Z, arma::mat &Ws, int k, int original_n,
						double C_trace, std::map<int, std::set<int>> &ml_map,
						std::vector<std::pair<int, int>> &local_cl_pairs,
						double ub, arma::sp_mat &node_assignment_X,
						arma::mat &node_centroids);

double ilp_heuristic(arma::mat &Z, arma::mat &Ws, int k, int original_n,
					double C_trace, arma::mat &init_centroids,
					std::map<int, std::set<int>> &ml_map,
					std::vector<std::pair<int, int>> &local_cl_pairs, double best_ub,
					arma::sp_mat &node_assignment_X, arma::mat &node_centroids);

double cluster_recovery(arma::mat &Ws, arma::mat &Ws_shr, arma::mat &Xopt, int k,
                        std::map<int, std::set<int>> &ml_map, std::vector<std::pair<int, int>> &local_cl,
                        std::vector<std::pair<int, int>> &global_ml, std::vector<std::pair<int, int>> &global_cl,
                        arma::sp_mat &assignment_X, arma::mat &out_centroids);

double get_feasible_clustering(arma::mat &Z, arma::mat &Ws, arma::mat &node_Ws, int k, double C_trace,
								std::vector<std::pair<int, int>> &global_ml_pairs,
								std::vector<std::pair<int, int>> &global_cl_pairs,
								std::map<int, std::set<int>> &ml_map,
								std::vector<std::pair<int, int>> &local_cl_pairs,
								arma::sp_mat &node_assignment_X, arma::mat &node_centroids);

double get_clustering_value(arma::mat &Z, arma::mat &Ws, int k, double C_trace,
							int original_n, std::map<int, std::set<int>> &ml_map,
							std::vector<std::pair<int, int>> &local_cl_pairs,
							arma::sp_mat &assignment_X, arma::mat &centroids);

#endif
