#include <iostream>
#include <map>
#include <algorithm>
#include <armadillo>
#include "sdp_branch_and_bound.h"

// data full path
const char *data_path;
const char *constraints_path;
const char *log_path;
const char *result_path;
std::ofstream log_file;

// branch and bound
double branch_and_bound_tol;
int branch_and_bound_parallel;
int branch_and_bound_max_nodes;
int branch_and_bound_visiting_strategy;

// sdp solver
int sdp_solver_session_threads_root;
int sdp_solver_session_threads;
const char *sdp_solver_folder;
double sdp_solver_tol;
int sdp_solver_stopoption;
int sdp_solver_maxiter;
int sdp_solver_maxtime;
int sdp_solver_verbose;
int sdp_solver_max_cp_iter_root;
int sdp_solver_max_cp_iter;
double sdp_solver_cp_tol;
int sdp_solver_max_ineq;
double sdp_solver_inherit_perc;
double sdp_solver_eps_ineq;
double sdp_solver_eps_active;
int sdp_solver_max_pair_ineq;
double sdp_solver_pair_perc;
int sdp_solver_max_triangle_ineq;
double sdp_solver_triangle_perc;

// heuristic
bool kmeans_sdp_based;
int kmeans_max_iter;
int kmeans_n_start;
int kmeans_permutations;
bool kmeans_verbose;

// read data Ws
arma::mat read_data(const char *filename, int &n, int &d, int &k) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << strerror(errno) << "\n";
        exit(EXIT_FAILURE);
    }
    // read the header n, d, k
    file >> n >> d >> k;
    arma::mat Ws(n, d);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            file >> Ws(i, j);
        }
    }
    return Ws;
}

void read_constraints(const char *filename, int n, UserConstraints &constraints){

	double gamma, delta;
	gamma = delta = -1;

    std::ifstream file(filename);
    if (!file) {
        std::cerr << strerror(errno) << "\n";
        exit(EXIT_FAILURE);
    }
	while (file.peek() != EOF){
		std::string constraint_type;
		file >> constraint_type;
		if (constraint_type == "ML" || constraint_type == "CL"){
			int a, b;
			file >> a >> b;
			if (a < 0 || a >= n || b < 0 || b >= n){
				std::cerr << "Invalid index in " << constraint_type << "\n";
				exit(EXIT_FAILURE);
			}
			std::pair<int,int> ab_pair(a,b);
			if (constraint_type == "ML")
				constraints.ml_pairs.push_back(ab_pair);
			else
				constraints.cl_pairs.push_back(ab_pair);

		} else if (constraint_type == "GAMMA"){
			file >> gamma;
		} else if (constraint_type == "DELTA"){
			file >> delta;
		} else {
			std::cerr << "Unknown constraint type " << constraint_type << "\n";
			exit(EXIT_FAILURE);
		}
		file.ignore();
	}
	if (delta == -1)
		delta = 0;
	if (gamma == -1)
		gamma = std::numeric_limits<double>::infinity();
	constraints.delta = delta;
	constraints.gamma = gamma;
}

std::map<std::string, std::string> read_params(std::string &config_file) {

    std::map<std::string, std::string> config_map = {};

    std::ifstream cFile (config_file);
    if (cFile.is_open()) {
        std::string line;
        while (getline(cFile, line)){
            line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
            if(line[0] == '#' || line.empty())
                continue;
            auto delimiterPos = line.find('=');
            auto key = line.substr(0, delimiterPos);
            auto value = line.substr(delimiterPos + 1);
            config_map.insert(std::pair<std::string, std::string>(key, value));
        }

    }
    else {
        std::cerr << "Couldn't open config file for reading.\n";
    }

    return config_map;
}


void run(int argc, char **argv) {

    std::string config_file = "config.txt";
    std::map<std::string, std::string> config_map = read_params(config_file);

    if (argc != 6) {
        std::cerr << "Input: <DATA_PATH> <K> <CONSTRAINTS_PATH> <LOG_PATH> <RESULT_PATH>" << std::endl;
        exit(EXIT_FAILURE);
    }

    data_path = argv[1];
    int k = std::stoi(argv[2]);
    constraints_path = argv[3];
    log_path = argv[4];
    result_path = argv[5];
    log_file.open(log_path);

    // branch and bound
    branch_and_bound_tol = std::stod(config_map["BRANCH_AND_BOUND_TOL"]);
    branch_and_bound_parallel = std::stoi(config_map["BRANCH_AND_BOUND_PARALLEL"]);
    branch_and_bound_max_nodes = std::stoi(config_map["BRANCH_AND_BOUND_MAX_NODES"]);
    branch_and_bound_visiting_strategy = std::stoi(config_map["BRANCH_AND_BOUND_VISITING_STRATEGY"]);

    // sdp solver
    // sdp_solver_matlab_session = config_map["SDP_SOLVER_MATLAB_SESSION"].c_str();
    sdp_solver_session_threads_root = std::stoi(config_map["SDP_SOLVER_SESSION_THREADS_ROOT"]);
    sdp_solver_session_threads = std::stoi(config_map["SDP_SOLVER_SESSION_THREADS"]);
    sdp_solver_folder = config_map["SDP_SOLVER_FOLDER"].c_str();
    sdp_solver_tol = std::stod(config_map["SDP_SOLVER_TOL"]);
    sdp_solver_verbose = std::stoi(config_map["SDP_SOLVER_VERBOSE"]);
    sdp_solver_max_cp_iter_root = std::stoi(config_map["SDP_SOLVER_MAX_CP_ITER_ROOT"]);
    sdp_solver_max_cp_iter = std::stoi(config_map["SDP_SOLVER_MAX_CP_ITER"]);
    sdp_solver_cp_tol = std::stod(config_map["SDP_SOLVER_CP_TOL"]);
    sdp_solver_max_ineq = std::stoi(config_map["SDP_SOLVER_MAX_INEQ"]);
    sdp_solver_inherit_perc = std::stod(config_map["SDP_SOLVER_INHERIT_PERC"]);
    sdp_solver_eps_ineq = std::stod(config_map["SDP_SOLVER_EPS_INEQ"]);
    sdp_solver_eps_active = std::stod(config_map["SDP_SOLVER_EPS_ACTIVE"]);
    sdp_solver_max_pair_ineq = std::stoi(config_map["SDP_SOLVER_MAX_PAIR_INEQ"]);
    sdp_solver_pair_perc = std::stod(config_map["SDP_SOLVER_PAIR_PERC"]);
    sdp_solver_max_triangle_ineq = std::stoi(config_map["SDP_SOLVER_MAX_TRIANGLE_INEQ"]);
    sdp_solver_triangle_perc = std::stod(config_map["SDP_SOLVER_TRIANGLE_PERC"]);
	sdp_solver_stopoption = 0;
	sdp_solver_maxiter = 50000;
	sdp_solver_maxtime = 3600;

    // kmeans
    kmeans_max_iter = 100;
    kmeans_n_start = 200;
    kmeans_verbose = 0;

    int n, d, old_k;
    arma::mat Ws = read_data(data_path, n, d, old_k);

	UserConstraints constraints;
	read_constraints(constraints_path, n, constraints);

    log_file << "\n" << "DATA_PATH, n, d, k: " << data_path << " " << n << " " << d << " " << k << "\n";
    log_file << "CONSTRAINTS_PATH: " << constraints_path << "\n";
    log_file << "LOG_PATH: " << log_path << "\n\n";

    log_file << "BRANCH_AND_BOUND_TOL: " << branch_and_bound_tol << "\n";
    log_file << "BRANCH_AND_BOUND_PARALLEL: " << branch_and_bound_parallel << "\n";
    log_file << "BRANCH_AND_BOUND_MAX_NODES: " <<  branch_and_bound_max_nodes << "\n";
    log_file << "BRANCH_AND_BOUND_VISITING_STRATEGY: " << branch_and_bound_visiting_strategy << "\n\n";

    log_file << "SDP_SOLVER_SESSION_THREADS_ROOT: " << sdp_solver_session_threads_root << "\n";
    log_file << "SDP_SOLVER_SESSION_THREADS: " << sdp_solver_session_threads << "\n";
    log_file << "SDP_SOLVER_FOLDER: " << sdp_solver_folder << "\n";
    log_file << "SDP_SOLVER_TOL: " << sdp_solver_tol << "\n";
    log_file << "SDP_SOLVER_VERBOSE: " << sdp_solver_verbose << "\n";
    log_file << "SDP_SOLVER_MAX_CP_ITER_ROOT: " << sdp_solver_max_cp_iter_root << "\n";
    log_file << "SDP_SOLVER_MAX_CP_ITER: " << sdp_solver_max_cp_iter << "\n";
    log_file << "SDP_SOLVER_CP_TOL: " << sdp_solver_cp_tol << "\n";
    log_file << "SDP_SOLVER_MAX_INEQ: " << sdp_solver_max_ineq << "\n";
    log_file << "SDP_SOLVER_INHERIT_PERC: " << sdp_solver_inherit_perc << "\n";
    log_file << "SDP_SOLVER_EPS_INEQ: " << sdp_solver_eps_ineq << "\n";
    log_file << "SDP_SOLVER_EPS_ACTIVE: " << sdp_solver_eps_active << "\n";
    log_file << "SDP_SOLVER_MAX_PAIR_INEQ: " << sdp_solver_max_pair_ineq << "\n";
    log_file << "SDP_SOLVER_PAIR_PERC: " << sdp_solver_pair_perc << "\n";
    log_file << "SDP_SOLVER_MAX_TRIANGLE_INEQ: " << sdp_solver_max_triangle_ineq << "\n";
    log_file << "SDP_SOLVER_TRIANGLE_PERC: " << sdp_solver_triangle_perc << "\n\n";

    sdp_branch_and_bound(k, Ws, constraints);

}

int main(int argc, char **argv) {

    run(argc, argv);

    return EXIT_SUCCESS;
}
