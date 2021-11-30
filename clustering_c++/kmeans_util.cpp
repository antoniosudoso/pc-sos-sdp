#include "kmeans_util.h"

void print_pairs(std::vector<std::pair<int, int>> &cl_vector) {
    for (auto &elem : cl_vector) {
        std::cout << "(" << elem.first << " " << elem.second << ")" << " ";
    }
    std::cout << "\n";
}

// sort the vector elements by second element of pairs
bool sort_by_value(const std::pair<int, double> &a, const std::pair<int, double> &b) {
    return (a.second < b.second);
}

// compute the l2-norm
double squared_distance(const arma::vec &a, const arma::vec &b) {
    double norm = arma::norm(a - b, 2);
    return std::pow(norm, 2);
}

arma::mat compute_distances(arma::mat &data){

	int n = data.n_rows;
	arma::mat distances = arma::zeros(n, n);

	double dist;
	for (int i = 0; i < n; i++){
		arma::vec point_i = data.row(i).t();
		for (int j = i+1; j < n; j++){
			arma::vec point_j = data.row(j).t();
			dist = squared_distance(point_i, point_j);
			distances(i, j) = dist;
			distances(j, i) = dist;
		}
	}
	return distances;
}


void add_both(std::map<int, std::set<int>> &graph, int i, int j) {
    graph[i].insert(j);
    graph[j].insert(i);
}

void dfs(int i, std::map<int, std::set<int>> &graph, std::vector<bool> &visited, std::vector<int> &component) {
    visited[i] = true;
    for (auto &j : graph[i]) {
        if (!visited[j]) {
            dfs(j, graph, visited, component);
        }
    }
    component.push_back(i);
}


LinkConstraint transitive_closure(std::vector<std::pair<int, int>> &ml, std::vector<std::pair<int, int>> &cl, int n) {

    std::map<int, std::set<int>> ml_graph;
    std::map<int, std::set<int>> cl_graph;

    for (int i = 0; i < n; i++) {
        ml_graph.insert(std::pair<int, std::set<int>> (i, {}));
        cl_graph.insert(std::pair<int, std::set<int>> (i, {}));
    }

    for (auto &pair_ml : ml) {
        add_both(ml_graph, pair_ml.first, pair_ml.second);
    }

    std::vector<bool> visited(n, false);
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            std::vector<int> component;
            dfs(i, ml_graph, visited, component);
            for (auto &x1 : component) {
                for (auto &x2 : component) {
                    if (x1 != x2) {
                        ml_graph[x1].insert(x2);
                    }
                }
            }
        }
    }

    for (auto &pair_cl : cl) {
        int i = pair_cl.first;
        int j = pair_cl.second;
        add_both(cl_graph, i, j);
        for (auto &y : ml_graph[j]) {
            add_both(cl_graph, i, y);
        }
        for (auto &x : ml_graph[i]) {
            add_both(cl_graph, x, j);
            for (auto &y : ml_graph[j]) {
                add_both(cl_graph, x, y);
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (auto &j : ml_graph[i]) {
            std::set<int> set_j = cl_graph[i];
            if (j != i) {
                if (set_j.count(j)) {
                    std::fprintf(stderr, "Inconsistent constraints between %d and %d", i, j);
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    return LinkConstraint{ml_graph, cl_graph};

}

void display_graph(std::map<int, std::set<int>> &map) {

    for (auto &map_elem : map) {
        int key = map_elem.first;
        std::set<int> value = map_elem.second;
        if (value.empty())
            continue;
        std::printf("%d: ", key);
        std::printf("{");
        for (auto &set_elem : value) {
            std::printf(" %d ", set_elem);
        }
        std::printf("}\n");
    }
}

/*
arma::vec closest_clusters(arma::mat &centers, arma::vec &point) {
    int n_centers = centers.n_rows;
    arma::vec distances(n_centers);
    for (int j = 0; j < n_centers; j++) {
        arma::vec center = centers.row(j).t();
        distances(j) = squared_distance(center, point);
    }
    return distances;
}
*/

arma::vec closest_clusters(std::vector<int> &centroids, int point, arma::mat &dist) {
    int n_centers = centroids.size();
    arma::vec distances(n_centers);
    for (int j = 0; j < n_centers; j++) {
        distances(j) = dist(centroids[j], point);
    }
    return distances;
}

// find points degree in the CL graph
std::vector<int> get_points_degrees(int n, std::map<int, std::set<int>> &ml_map,
									std::vector<std::pair<int,int>> &cl_pairs){

	std::vector<int> local_degrees = std::vector<int>(ml_map.size(), 0);
	std::vector<int> degrees;
	degrees.resize(n);

	for (auto &pair : cl_pairs){
		local_degrees[pair.first] += 1;
		local_degrees[pair.second] += 1;
	}
	for (int i = 0; i < ml_map.size(); i++){
		for (int j : ml_map[i]){
			degrees[j] = local_degrees[i];
		}
	}
	return degrees;
}

// sort points according to their degree in the CL graph
std::vector<int> get_points_sorting(std::vector<int> degrees){

	std::vector<int> sorting;
	sorting.resize(degrees.size());

	std::size_t c(0);
	std::generate(std::begin(sorting), std::end(sorting), [&]{ return c++; });

	std::sort(std::begin(sorting), std::end(sorting),
			  [&](int i1, int i2) { return degrees[i1] > degrees[i2]; });
	return sorting;
}

std::vector<int> get_weighted_permutation(std::vector<int> &degrees, std::mt19937 gen) {
	
	int n = degrees.size();
	std::vector<int> permutation;
	permutation.resize(n);

	int non_zeros = 0;
    arma::vec chances = arma::zeros(n);
    for (int i = 0; i < n; i++){
		if ((chances(i) += degrees[i]) > 0)
			non_zeros++;
	}
	
    std::uniform_real_distribution<double> dis(0.0, 1.0);
	
	int i;
    for (i = 0; i < non_zeros; i++) {

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
		permutation[i] = index;
		chances(index) = 0;
    }
	int j = 0;
	for ( ; i < n; i++){
		while (degrees[j] > 0) j++;
		permutation[i] = j;
		j++;
	}

	return permutation;
}
