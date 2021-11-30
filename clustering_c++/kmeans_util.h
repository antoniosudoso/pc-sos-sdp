#ifndef CLUSTERING_KMEANS_UTIL_H
#define CLUSTERING_KMEANS_UTIL_H

#include <armadillo>
#include <set>

typedef struct LinkConstraint {

    // must-link
    std::map<int, std::set<int>> ml_graph;
    // cannot-link
    std::map<int, std::set<int>> cl_graph;

} LinkConstraint;

bool sort_by_value(const std::pair<int, double> &a, const std::pair<int, double> &b);
double squared_distance(const arma::vec &a, const arma::vec &b);
arma::mat compute_distances(arma::mat &data);
LinkConstraint transitive_closure(std::vector<std::pair<int, int>> &ml, std::vector<std::pair<int, int>> &cl, int n);
void display_graph(std::map<int, std::set<int>> &map);
void print_pairs(std::vector<std::pair<int, int>> &cl_vector);
void add_both(std::map<int, std::set<int>> &graph, int i, int j);
void dfs(int i, std::map<int, std::set<int>> &graph, std::vector<bool> &visited, std::vector<int> &component);
arma::vec closest_clusters(std::vector<int> &centroids, int point, arma::mat &dist);
std::vector<int> get_points_degrees(int n, std::map<int, std::set<int>> &ml_map,
									std::vector<std::pair<int,int>> &cl_pairs);
std::vector<int> get_points_sorting(std::vector<int> degrees);
std::vector<int> get_weighted_permutation(std::vector<int> &degrees, std::mt19937 gen);

#endif //CLUSTERING_KMEANS_UTIL_H
