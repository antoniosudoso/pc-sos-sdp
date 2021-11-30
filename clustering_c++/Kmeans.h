#ifndef CLUSTERING_KMEANS_H
#define CLUSTERING_KMEANS_H

#include <armadillo>
#include "kmeans_util.h"

class Kmeans {

private:

    bool verbose;
    arma::mat data;
    arma::mat centroids;
    arma::vec assignments;
    int n, d, k;
    double loss;
    LinkConstraint constraint;
	std::vector<int> cl_degrees;
	std::vector<int> points_sorting;

    double objectiveFunction();
    bool assignPoints(std::vector<int> &permutation);
    bool computeCentroids();
    void initCentroids(arma::mat &distances);
    bool violateConstraint(int point_i, int cluster_j);

public:

    Kmeans(const arma::mat &data, int k, std::map<int, std::set<int>> &ml_map,
					std::vector<std::pair<int, int>> &local_cl_pair,
					std::vector<std::pair<int, int>> &global_ml,
					std::vector<std::pair<int, int>> &global_cl, bool verbose);
	Kmeans(const arma::mat &data, int k, std::vector<std::pair<int, int>> &global_ml,
					std::vector<std::pair<int, int>> &global_cl, bool verbose);
    bool start(int max_iter, int n_start, int n_permutations);
    bool start(int max_iter, int n_start, int n_permutations, arma::mat &distances);
    bool start(int max_iter, int n_permutations,  arma::mat &init_centroids);
	bool findClustering(int n_start, int n_permutations, arma::mat distances);
    arma::sp_mat getAssignments();
    double getLoss();
    arma::mat getCentroids();

};



#endif //CLUSTERING_KMEANS_H
