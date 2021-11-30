#include <thread>
#include "matlab_util.h"
#include "sdp_branch_and_bound.h"
#include "sdp_solver_util.h"
#include "JobQueue.h"
#include "util.h"
#include "config_params.h"
#include "Node.h"
#include "ThreadPool.h"
#include "feasibility.h"
#include "ub_heuristics.h"

void save_X_to_file(arma::sp_mat &X){

	std::ofstream f;
	f.open(result_path);
	for (int i = 0; i < X.n_rows; i++){
		int val = X(i,0);
		f << val;
		for (int j = 1; j < X.n_cols; j++){
			val = X(i,j);
			f << " " << val;
		}
		f << "\n";
	}
	f.close();
}

// root
SDPResult solve_sdp(std::unique_ptr<matlab::engine::MATLABEngine> &matlabPtr, matlab::data::ArrayFactory &factory,
                    arma::mat &C, arma::sp_mat &A, arma::vec &b, int k,
					int original_n, double original_trace, double global_ub) {

    // convert data
    matlab::data::TypedArray<double> C_matlab = arma_to_matlab_matrix(factory, C);
    matlab::data::CellArray A_matlab = arma_to_matlab_cell(factory, A);
    matlab::data::TypedArray<double> b_matlab = arma_to_matlab_vector(factory, b);

    // Pass vector containing args in vector
    std::vector<matlab::data::Array> args({
        factory.createScalar<int>(sdp_solver_session_threads_root),
        C_matlab, A_matlab, b_matlab,
        factory.createScalar<double>(k),
        factory.createScalar<double>(original_n),
        factory.createScalar<double>(original_trace),
        factory.createScalar<int>(sdp_solver_stopoption),
        factory.createScalar<double>(sdp_solver_maxiter),
        factory.createScalar<int>(sdp_solver_maxtime),
        factory.createScalar<double>(sdp_solver_tol),
        factory.createScalar<int>(sdp_solver_verbose),
        factory.createScalar<int>(sdp_solver_max_cp_iter_root),
        factory.createScalar<double>(sdp_solver_cp_tol),
        factory.createScalar<double>(global_ub),
        factory.createScalar<double>(branch_and_bound_tol),
        factory.createScalar<double>(sdp_solver_eps_ineq),
        factory.createScalar<double>(sdp_solver_eps_active),
        factory.createScalar<int>(sdp_solver_max_ineq),
        factory.createScalar<int>(sdp_solver_max_pair_ineq),
        factory.createScalar<double>(sdp_solver_pair_perc),
        factory.createScalar<int>(sdp_solver_max_triangle_ineq),
        factory.createScalar<double>(sdp_solver_triangle_perc)});

    // Call MATLAB function and return result
    const size_t n_return = 11;
    matlabPtr->eval(u"clear");
    std::vector<matlab::data::Array> result = matlabPtr->feval(u"solve_cluster_cp", n_return, args);

    matlab::data::Array bound =  result[0];
    matlab::data::TypedArray<double> X_matlab = result[1];
    matlab::data::Array flag =  result[2];
    matlab::data::Array ineq = result[3];
    matlab::data::Array iter = result[4];
    matlab::data::Array iter_flag = result[5];
    matlab::data::Array pair = result[6];
    matlab::data::Array triangle = result[7];
    matlab::data::Array clique = result[8];
    matlab::data::CellArray B_matlab = result[9];
    matlab::data::TypedArray<double> l_matlab = result[10];

    double lower_bound = (double) bound[0];
    arma::mat X = matlab_to_arma_matrix(X_matlab);
    int info_flag = (int) flag[0];
    int cp_iter = (int) iter[0];
    int cp_flag = (int) iter_flag[0];
    double n_pair = (double) pair[0];
    double n_triangle = (double) triangle[0];
    double n_clique = (double) clique[0];
    int n_ineq = (int) ineq[0];
    std::vector<arma::sp_mat> B_vector = matlab_to_arma_sp_mat_vector(B_matlab);
    arma::vec l_vec = matlab_to_arma_vector(l_matlab);
    return SDPResult{info_flag, X, lower_bound, n_ineq, cp_iter, cp_flag, n_pair, n_triangle, n_clique, B_vector, l_vec};
}

// lower bound must link and cannot link
SDPResult solve_sdp(std::unique_ptr<matlab::engine::MATLABEngine> &matlabPtr, matlab::data::ArrayFactory &factory,
        arma::mat &C, arma::sp_mat &A, arma::vec &b, int k, int original_n, double original_trace, double global_ub,
        std::vector<arma::sp_mat> &parent_B_vector, arma::vec &parent_l_vec, int parent_n, int i, int j) {

    // convert data
    matlab::data::TypedArray<double> C_matlab = arma_to_matlab_matrix(factory, C);
    matlab::data::CellArray A_matlab = arma_to_matlab_cell(factory, A);
    matlab::data::TypedArray<double> b_matlab = arma_to_matlab_vector(factory, b);
    matlab::data::CellArray parent_Bcell = arma_to_matlab_cell(factory, parent_B_vector);
    matlab::data::TypedArray<double> parent_l = arma_to_matlab_vector(factory, parent_l_vec);

    // Pass vector containing args in vector
    std::vector<matlab::data::Array> args({
        factory.createScalar<int>(sdp_solver_session_threads),
        C_matlab, A_matlab, b_matlab,
        factory.createScalar<double>(k),
        factory.createScalar<double>(original_n),
        factory.createScalar<double>(original_trace),
        factory.createScalar<int>(sdp_solver_stopoption),
        factory.createScalar<double>(sdp_solver_maxiter),
        factory.createScalar<int>(sdp_solver_maxtime),
        factory.createScalar<double>(sdp_solver_tol),
        factory.createScalar<int>(sdp_solver_verbose),
        factory.createScalar<int>(sdp_solver_max_cp_iter),
        factory.createScalar<double>(sdp_solver_cp_tol),
        factory.createScalar<double>(global_ub),
        factory.createScalar<double>(branch_and_bound_tol),
        factory.createScalar<double>(sdp_solver_eps_ineq),
        factory.createScalar<double>(sdp_solver_eps_active),
        factory.createScalar<int>(sdp_solver_max_ineq),
        factory.createScalar<int>(sdp_solver_max_pair_ineq),
        factory.createScalar<double>(sdp_solver_pair_perc),
        factory.createScalar<int>(sdp_solver_max_triangle_ineq),
        factory.createScalar<double>(sdp_solver_triangle_perc),
        parent_Bcell, parent_l,
        factory.createScalar<double>(parent_n),
        factory.createScalar<double>(sdp_solver_inherit_perc),
        factory.createScalar<int>(i),
        factory.createScalar<int>(j)});

    // Call MATLAB function and return result
    const size_t n_return = 11;
    matlabPtr->eval(u"clear");
    std::vector<matlab::data::Array> result = matlabPtr->feval(u"solve_cluster_cp_inherit", n_return, args);

    matlab::data::Array bound =  result[0];
    matlab::data::TypedArray<double> X_matlab = result[1];
    matlab::data::Array flag =  result[2];
    matlab::data::Array ineq = result[3];
    matlab::data::Array iter = result[4];
    matlab::data::Array iter_flag = result[5];
    matlab::data::Array pair = result[6];
    matlab::data::Array triangle = result[7];
    matlab::data::Array clique = result[8];
    matlab::data::CellArray B_matlab = result[9];
    matlab::data::TypedArray<double> l_matlab = result[10];

    double lower_bound = (double) bound[0];
    arma::mat X = matlab_to_arma_matrix(X_matlab);

    int info_flag = (int) flag[0];
    int cp_iter = (int) iter[0];
    int cp_flag = (int) iter_flag[0];
    double n_pair = (double) pair[0];
    double n_triangle = (double) triangle[0];
    double n_clique = (double) clique[0];
    int n_ineq = (int) ineq[0];
    std::vector<arma::sp_mat> B_vector = matlab_to_arma_sp_mat_vector(B_matlab);
    arma::vec l_vec = matlab_to_arma_vector(l_matlab);
    return SDPResult{info_flag, X, lower_bound, n_ineq, cp_iter, cp_flag, n_pair, n_triangle, n_clique, B_vector, l_vec};
}




std::pair<JobData *, JobData *> create_cl_ml_jobs(double node_gap, SDPNode *node, arma::mat &X,
                                                  NodeData *parent, SharedData *shared_data, InputData *input_data) {

	if (std::isinf(node->lb) || node_gap <= branch_and_bound_tol) {
        // std::cout << "PRUNING " << node->id << "\n";
        delete(node);
        if (parent != nullptr) {
            delete (parent->node);
            delete (parent);
        }
        return std::make_pair(nullptr, nullptr);
    }

    std::pair<int, int> var = find_branch_norm(X);

    int i = var.first;
    int j = var.second;

    if (i == -1 && j == -1) {

        const std::lock_guard<std::mutex> lock(shared_data->queueMutex);

        log_file << "PRUNING BY OPTIMALITY " << node->id << "\n";
        if (node->lb - shared_data->global_ub <= -branch_and_bound_tol) {
            // update global upper bound, run the heuristic instead of setting global_ub = node->lb
			arma::sp_mat assignment_X;
			arma::mat centroids;
			double new_ub = get_clustering_value(X, node->Ws, input_data->k,
												input_data->C_trace, input_data->Ws.n_rows,
												node->ml_map, node->local_cl_pairs,
												assignment_X, centroids);
			if (shared_data->global_ub > new_ub){
				shared_data->global_ub = new_ub;
				shared_data->global_X = assignment_X;
				shared_data->global_centroids = centroids;
			}
        }

        delete (node);
        if (parent != nullptr) {
            delete (parent->node);
            delete (parent);
        }
        return std::make_pair(nullptr, nullptr);

        // mutex is automatically released when lock goes out of scope

    }

    /*
    auto *copy_node_cl = new SDPNode(*node);
    auto *copy_node_ml = new SDPNode(*node);
    delete (node);
    */

    auto *cl_data = new NodeData();
    cl_data->node = new SDPNode(*node);
    cl_data->i = i;
    cl_data->j = j;

    auto *ml_data = new NodeData();
    ml_data->node = new SDPNode(*node);
    ml_data->i = i;
    ml_data->j = j;

    auto *cl_job_data = new JobData();
    cl_job_data->type = CANNOT_LINK;
    cl_job_data->node_data = cl_data;

    auto *ml_job_data = new JobData();
    ml_job_data->type = MUST_LINK;
    ml_job_data->node_data = ml_data;

    if (parent != nullptr) {
        delete (parent->node);
        delete (parent);
    }

    delete (node);

    return std::make_pair(cl_job_data, ml_job_data);

}


std::pair<JobData *, JobData *> build_cl_problem(MatlabStruct *matlab_struct, NodeData *node_data, InputData *input_data, SharedData  *shared_data) {


    // generate cannot link child
    auto cl_node = new SDPNode();
	auto parent = node_data->node;

	double parent_gap = (shared_data->global_ub - parent->lb) / shared_data->global_ub;
	if (parent_gap <= branch_and_bound_tol)
        return std::make_pair(nullptr, nullptr);

    cl_node->ml_map = parent->ml_map;
    cl_node->local_cl_pairs = parent->local_cl_pairs;
    cl_node->local_cl_pairs.emplace_back(node_data->i, node_data->j);
    cl_node->global_ml_pairs = build_global_must_link_pairs(cl_node->ml_map);
    cl_node->global_cl_pairs = build_global_cannot_link_pairs(cl_node->ml_map, cl_node->local_cl_pairs);
    // inherit Ws, A and add the cannot link constraint
    cl_node->Ws = parent->Ws;
    cl_node->A = build_A_cannot_link(parent->A, node_data->i, node_data->j);
    cl_node->b = build_b_cannot_link(parent->b);
    // cl_node->l = parent->l + 1;

	int n = cl_node->Ws.n_rows;

    auto start_time = std::chrono::high_resolution_clock::now();

    int flag;
    double n_pair, n_triangle, n_clique;
    int n_ineq;
    int cp_iter;
    int cp_flag;
	arma::mat X;
	
	int feasible;
	if (!std::isinf(parent->ub) &&
			!have_same_assignment(parent->assignment_X, node_data->i, node_data->j, parent->ml_map)){
		cl_node->ub = parent->ub;
		cl_node->assignment_X = parent->assignment_X;
		cl_node->centroids = parent->centroids;
		feasible = 1;
	} else {
		cl_node->ub = std::numeric_limits<double>::infinity();
		feasible = is_feasible(cl_node->Ws, input_data->k, cl_node->ml_map, cl_node->local_cl_pairs);
	}

	if (!feasible){

		flag = 2;
		n_pair = 0;
		n_triangle = 0;
		n_clique = 0;
		cp_iter = 0;
		cp_flag = CP_FLAG_INFEAS;
		n_ineq = 0;
		cl_node->lb = std::numeric_limits<double>::infinity();
		X = arma::zeros(0,0);

	} else {

		arma::mat C = cl_node->Ws * cl_node->Ws.t();
		SDPResult sdp_result = solve_sdp(matlab_struct->matlabPtr, matlab_struct->factory,
										 C, cl_node->A, cl_node->b, input_data->k, input_data->Ws.n_rows, input_data->C_trace,
										 shared_data->global_ub, parent->B_vector, parent->l_vec,
										 parent->Ws.n_rows, node_data->i, node_data->j);

		flag = sdp_result.flag;
		n_pair = sdp_result.n_pair;
		n_triangle = sdp_result.n_triangle;
		n_clique = sdp_result.n_clique;
		cp_iter = sdp_result.cp_iter;
		cp_flag = sdp_result.cp_flag;
		n_ineq = sdp_result.n_ineq;
		cl_node->lb = std::max(sdp_result.lb + input_data->C_trace, parent->lb);
		cl_node->B_vector = sdp_result.B_vector;
		cl_node->l_vec = sdp_result.l_vec;
		X = sdp_result.X;

		if (cl_node->lb <= shared_data->global_ub){
			double new_ub = ilp_heuristic(X, cl_node->Ws, input_data->k,
									input_data->Ws.n_rows, input_data->C_trace,
									cl_node->ml_map, cl_node->local_cl_pairs, cl_node->ub,
									cl_node->assignment_X, cl_node->centroids);
			cl_node->ub = std::min(cl_node->ub, new_ub);
		}
	}

    double node_gap;

    {
        const std::lock_guard<std::mutex> lock(shared_data->queueMutex);

        cl_node->id = shared_data->n_nodes;

        shared_data->n_nodes++;
        shared_data->sum_ineq += n_pair + n_triangle + n_clique;
        shared_data->sum_cp_iter += cp_iter;

        bool ub_updated = false;
        if (cl_node->ub - shared_data->global_ub <= -branch_and_bound_tol) {
            // update global upper bound
            shared_data->global_ub = cl_node->ub;
            shared_data->global_X = cl_node->assignment_X;
            shared_data->global_centroids = cl_node->centroids;
            ub_updated = true;
        }


        int open = shared_data->queue->getSize();

        node_gap = (shared_data->global_ub - cl_node->lb) / shared_data->global_ub;

        double gap = node_gap;
        Node *min_lb_node = shared_data->queue->getMinLb();
        if (min_lb_node != nullptr)
            gap = (shared_data->global_ub - min_lb_node->lb) / shared_data->global_ub;

        shared_data->gap = gap;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        double time = duration.count();

        //std::cout << '\r';
        print_log_sdp(log_file, cl_node->Ws.n_rows, parent->id, cl_node->id, parent->lb, cl_node->lb,
                      flag, time, cp_iter, cp_flag, n_ineq, n_pair, n_triangle, n_clique, cl_node->ub,
					  shared_data->global_ub, node_data->i, node_data->j, node_gap, shared_data->gap, open, ub_updated);

    }

    // mutex is automatically released when lock goes out of scope

    return create_cl_ml_jobs(node_gap, cl_node, X, node_data, shared_data, input_data);

}


std::pair<JobData *, JobData *> build_ml_problem(MatlabStruct *matlab_struct, NodeData *node_data, InputData *input_data, SharedData *shared_data) {

    // generate must link child
    SDPNode *ml_node;
	SDPNode *parent = node_data->node;

	double parent_gap = (shared_data->global_ub - parent->lb) / shared_data->global_ub;
	if (parent_gap <= branch_and_bound_tol)
        return std::make_pair(nullptr, nullptr);

    ml_node = new SDPNode();
    ml_node->Ws = build_Ws_must_link(parent->Ws, node_data->i, node_data->j);
    ml_node->ml_map = build_must_link_map(parent->ml_map, node_data->i, node_data->j);
    ml_node->local_cl_pairs = parent->local_cl_pairs;
    std::set<int> dup_indices = update_cannot_link(ml_node->local_cl_pairs, node_data->i, node_data->j);
    ml_node->global_cl_pairs = build_global_cannot_link_pairs(ml_node->ml_map, ml_node->local_cl_pairs);
    ml_node->global_ml_pairs = build_global_must_link_pairs(ml_node->ml_map);
    // ml_node->l = parent->l + 1;

    int n = parent->Ws.n_rows;
    if (n - 1 == input_data->k) {
        // build directly the assignment matrix
        // ml_node->l = parent->l + 1;
        ml_node->ub = build_X_from_ml(input_data->Ws, ml_node->ml_map, ml_node->assignment_X);
        ml_node->lb = ml_node->ub;

        const std::lock_guard<std::mutex> lock(shared_data->queueMutex);

        ml_node->id = shared_data->n_nodes;
        shared_data->n_nodes++;

        int open = shared_data->queue->getSize();

        bool ub_updated = false;
        if (ml_node->ub - shared_data->global_ub <= -branch_and_bound_tol) {
            // update global upper bound
            shared_data->global_ub = ml_node->ub;
            shared_data->global_X = ml_node->assignment_X;
            shared_data->global_centroids = ml_node->centroids;
            ub_updated = true;
        }

        double node_gap = (shared_data->global_ub - ml_node->lb) / shared_data->global_ub;

        double gap = node_gap;
        Node *min_lb_node = shared_data->queue->getMinLb();
        if (min_lb_node != nullptr)
            gap = (shared_data->global_ub - min_lb_node->lb) / shared_data->global_ub;

        shared_data->gap = gap;

        print_log_sdp(log_file, ml_node->Ws.n_rows, parent->id, ml_node->id, parent->lb,
                      ml_node->lb,0, 0, 0, 0, 0, 0, 0, 0, ml_node->ub,
                      shared_data->global_ub, node_data->i, node_data->j, node_gap, shared_data->gap, open, ub_updated);

        delete (ml_node);
        delete (parent);
        return std::make_pair(nullptr, nullptr);

    }

    arma::sp_mat TTt = build_TTt(ml_node->ml_map);
    ml_node->A = build_A_must_link(n, TTt);
    ml_node->b = build_b_must_link(n, input_data->k);
    for (auto &elem : ml_node->local_cl_pairs) {
        ml_node->A = build_A_cannot_link(ml_node->A, elem.first, elem.second);
        ml_node->b = build_b_cannot_link(ml_node->b);
    }
	n--;

    auto start_time = std::chrono::high_resolution_clock::now();

    int flag;
    double n_pair, n_triangle, n_clique;
    int n_ineq;
    int cp_iter;
    int cp_flag;
	arma::mat X;

	int feasible;
	if (!std::isinf(parent->ub) &&
			have_same_assignment(parent->assignment_X, node_data->i, node_data->j, parent->ml_map)){
		ml_node->ub = parent->ub;
		ml_node->assignment_X = parent->assignment_X;
		ml_node->centroids = parent->centroids;
		feasible = 1;
	} else {
		ml_node->ub = std::numeric_limits<double>::infinity();
		feasible = is_feasible(ml_node->Ws, input_data->k, ml_node->ml_map, ml_node->local_cl_pairs);
	}

	if (!feasible){

		flag = 2;
		n_pair = 0;
		n_triangle = 0;
		n_clique = 0;
		cp_iter = 0;
		cp_flag = CP_FLAG_INFEAS;
		n_ineq = 0;
		ml_node->lb = std::numeric_limits<double>::infinity();
		X = arma::zeros(0,0);

	} else {

		arma::mat C = ml_node->Ws * ml_node->Ws.t();
		SDPResult sdp_result = solve_sdp(matlab_struct->matlabPtr, matlab_struct->factory,
										 C, ml_node->A, ml_node->b, input_data->k, input_data->Ws.n_rows, input_data->C_trace,
										 shared_data->global_ub, parent->B_vector, parent->l_vec,
										 parent->Ws.n_rows, node_data->i, node_data->j);

		flag = sdp_result.flag;
		n_pair = sdp_result.n_pair;
		n_triangle = sdp_result.n_triangle;
		n_clique = sdp_result.n_clique;
		cp_iter = sdp_result.cp_iter;
		cp_flag = sdp_result.cp_flag;
		n_ineq = sdp_result.n_ineq;
		ml_node->lb = std::max(sdp_result.lb + input_data->C_trace, parent->lb);
		ml_node->B_vector = sdp_result.B_vector;
		ml_node->l_vec = sdp_result.l_vec;
		X = sdp_result.X;

		if (ml_node->lb <= shared_data->global_ub){
			double new_ub = ilp_heuristic(X, ml_node->Ws, input_data->k,
									input_data->Ws.n_rows, input_data->C_trace,
									ml_node->ml_map, ml_node->local_cl_pairs, ml_node->ub,
									ml_node->assignment_X, ml_node->centroids);
			ml_node->ub = std::min(ml_node->ub, new_ub);
		}
	}

    double node_gap;

    {
        const std::lock_guard<std::mutex> lock(shared_data->queueMutex);

        ml_node->id = shared_data->n_nodes;

        shared_data->n_nodes++;
        shared_data->sum_ineq += n_pair + n_triangle + n_clique;
        shared_data->sum_cp_iter += cp_iter;

        bool ub_updated = false;
        if (ml_node->ub - shared_data->global_ub <= -branch_and_bound_tol) {
            // update global upper bound
            shared_data->global_ub = ml_node->ub;
            shared_data->global_X = ml_node->assignment_X;
            shared_data->global_centroids = ml_node->centroids;
            ub_updated = true;
        }

        int open = shared_data->queue->getSize();

        node_gap = (shared_data->global_ub - ml_node->lb) / shared_data->global_ub;

        double gap = node_gap;
        Node *min_lb_node = shared_data->queue->getMinLb();
        if (min_lb_node != nullptr)
            gap = (shared_data->global_ub - min_lb_node->lb) / shared_data->global_ub;

        shared_data->gap = gap;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        double time = duration.count();

        //std::cout << '\r';
        print_log_sdp(log_file, ml_node->Ws.n_rows, parent->id, ml_node->id, parent->lb, ml_node->lb,
                      flag, time, cp_iter, cp_flag, n_ineq, n_pair, n_triangle, n_clique, ml_node->ub,
					  shared_data->global_ub, node_data->i, node_data->j, node_gap, shared_data->gap, open, ub_updated);

    }

    // mutex is automatically released when lock goes out of scope

    return create_cl_ml_jobs(node_gap, ml_node, X, node_data, shared_data, input_data);


}


int get_delta_gamma_constraints(arma::mat &data, double delta, double gamma,
							std::vector<std::pair<int,int>> &ml_pairs,
							std::vector<std::pair<int,int>> &cl_pairs){

	if (delta == 0 && std::isinf(gamma))
		return 0;

	int n = data.n_rows;
	double min_dist = std::numeric_limits<double>::infinity();
	double max_dist = 0;
	for (int i = 0; i < n; i++){
		for (int j = i+1; j < n; j++){
			double dij = std::sqrt(squared_distance(data.row(i).t(), data.row(j).t()));
			if (dij < min_dist)
				min_dist = dij;
			if (dij > max_dist)
				max_dist = dij;
			if (dij < delta)
				ml_pairs.push_back(std::pair<int,int>(i, j));
			else if (dij > gamma)
				cl_pairs.push_back(std::pair<int,int>(i, j));
		}
	}
	if (max_dist < delta) {
		log_file << "Delta constraint is infeasible\n";
		return -1;
	}
	if (min_dist > gamma) {
		log_file << "Gamma constraint is infeasible\n";
		return -1;
	}
	return 0;
}



std::map<int, std::set<int>> get_ml_map(int n, std::vector<std::pair<int, int>> &ml) {

    std::map<int, std::set<int>> ml_graph;
    std::map<int, std::set<int>> ml_map;

    for (int i = 0; i < n; i++) {
        ml_graph.insert(std::pair<int, std::set<int>> (i, {}));
    }

    for (auto &pair_ml : ml) {
        add_both(ml_graph, pair_ml.first, pair_ml.second);
    }

	int components_counter = 0;
    std::vector<bool> visited(n, false);
    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            std::vector<int> component;
            dfs(i, ml_graph, visited, component);
			std::set<int> component_set(component.begin(), component.end());
			ml_map[components_counter] = component_set;
			components_counter++;
        }
    }

	return ml_map;
}

int get_root_constraints(arma::mat &Ws, int k, UserConstraints &constraints,
							std::vector<std::pair<int,int>> &global_ml_pairs,
							std::vector<std::pair<int,int>> &global_cl_pairs,
							std::map<int, std::set<int>> &ml_map,
							std::vector<std::pair<int,int>> &local_cl_pairs){

	int n = Ws.n_rows;

	if (constraints.gamma < constraints.delta){
		log_file << "Infeasibility: Gamma and delta constraints are infeasible\n";
		return -1;
	}	

	if (get_delta_gamma_constraints(Ws, constraints.delta, constraints.gamma,
							constraints.ml_pairs, constraints.cl_pairs) != 0){
		return -1;
	}

	global_ml_pairs = constraints.ml_pairs;
	global_cl_pairs = constraints.cl_pairs;

	if (k == 2){
		std::vector<std::vector<int>> cl_neighbors(n);
		for (auto &cl_pair : global_cl_pairs){
			int a = cl_pair.first;
			int b = cl_pair.second;
			cl_neighbors[a].push_back(b);
			cl_neighbors[b].push_back(a);
		}
		for (int i = 0; i < n; i++){
			if (cl_neighbors[i].size() >= 2){
				for (int j = 1; j < cl_neighbors[i].size(); j++){
					global_ml_pairs.push_back(std::pair<int,int>(cl_neighbors[i][0], cl_neighbors[i][j]));
				}
			}
		}
	}

	ml_map = get_ml_map(n, global_ml_pairs);

	int a, b, block_a, block_b;
	std::vector<int> blocks(n);

	for (int block = 0; block < ml_map.size(); block++){
		for (auto x : ml_map[block]){
			blocks[x] = block;
		}
	}

	std::set<std::pair<int,int>> cl_pairs_set;
	for (auto &cl_pair : global_cl_pairs){
		a = cl_pair.first;
		b = cl_pair.second;
		block_a = blocks[a];
		block_b = blocks[b];

		if (block_a == block_b){
			std::cout << "Infeasibility: Must-link and Cannot-link on (" << a << ", " << b << ")\n";
			return -1;
		} else if (block_a < block_b){
			cl_pairs_set.insert(std::pair<int,int>(block_a, block_b));
		} else {
			cl_pairs_set.insert(std::pair<int,int>(block_b, block_a));
		}
	}
	local_cl_pairs.assign(cl_pairs_set.begin(), cl_pairs_set.end());
	return 0;	
}


std::pair<JobData *, JobData *> build_root_problem(MatlabStruct *matlab_struct,
								InputData *input_data, SharedData *shared_data,
												UserConstraints &constraints) {

    // number of data points
    int n = input_data->Ws.n_rows;
    // init root
    SDPNode *root;
    root = new SDPNode();
    root->id = shared_data->n_nodes;
    //root->l = 0;

	if (get_root_constraints(input_data->Ws, input_data->k, constraints,
						root->global_ml_pairs, root->global_cl_pairs,
						root->ml_map, root->local_cl_pairs) != 0){
        return std::make_pair(nullptr, nullptr);
	}

	n = root->ml_map.size();
	if (n < input_data->k){
		log_file << "Infeasibility: k is too large\n";
        return std::make_pair(nullptr, nullptr);
	}	

    root->Ws = build_Ws(input_data->Ws, root->ml_map);
    root->A = build_A(root->ml_map, root->local_cl_pairs);
    root->b = build_b(input_data->k, root->ml_map, root->local_cl_pairs);
	
    double C_trace = input_data->C_trace;

    auto start_time = std::chrono::high_resolution_clock::now();

    int flag;
    double n_pair, n_triangle, n_clique;
    int n_ineq;
    int cp_iter;
    int cp_flag;
	arma::mat X;

	root->ub = std::numeric_limits<double>::infinity();
	int feasible = is_feasible(root->Ws, input_data->k, root->ml_map, root->local_cl_pairs);

	if (!feasible){

		flag = 2;
		n_pair = 0;
		n_triangle = 0;
		n_clique = 0;
		cp_iter = 0;
		cp_flag = CP_FLAG_INFEAS;
		n_ineq = 0;
		root->lb = std::numeric_limits<double>::infinity();
		X = arma::zeros(0,0);

	} else {

		arma::mat C = root->Ws * root->Ws.t();
		SDPResult sdp_result = solve_sdp(matlab_struct->matlabPtr, matlab_struct->factory,
									 C, root->A, root->b, input_data->k, input_data->Ws.n_rows,
									 input_data->C_trace, shared_data->global_ub);

		flag = sdp_result.flag;
		n_pair = sdp_result.n_pair;
		n_triangle = sdp_result.n_triangle;
		n_clique = sdp_result.n_clique;
		cp_iter = sdp_result.cp_iter;
		cp_flag = sdp_result.cp_flag;
		n_ineq = sdp_result.n_ineq;
		root->lb = sdp_result.lb + C_trace;
		root->B_vector = sdp_result.B_vector;
		root->l_vec  = sdp_result.l_vec;
		X = sdp_result.X;

		double new_ub = ilp_heuristic(X, root->Ws, input_data->k,
								input_data->Ws.n_rows, input_data->C_trace,
								root->ml_map, root->local_cl_pairs, root->ub,
								root->assignment_X, root->centroids);
		root->ub = std::min(root->ub, new_ub);
	}

    shared_data->global_ub = root->ub;
    shared_data->global_X = root->assignment_X;
	shared_data->global_centroids = root->centroids;

    shared_data->n_nodes++;
    shared_data->sum_cp_iter += cp_iter;
    shared_data->sum_ineq += n_pair + n_triangle + n_clique;

    int open = shared_data->queue->getSize();

    double node_gap = (shared_data->global_ub - root->lb) / shared_data->global_ub;
    shared_data->gap = node_gap;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    double time = duration.count();

    // std::cout << '\r';
    print_log_sdp(log_file, n, -1, root->id, -std::numeric_limits<double>::infinity(), root->lb,
                  flag, time, cp_iter, cp_flag, n_ineq, n_pair, n_triangle, n_clique, root->ub,
				  shared_data->global_ub, -1, -1, node_gap, node_gap, open, true);

    return create_cl_ml_jobs(node_gap, root, X, nullptr, shared_data, input_data);

}

bool is_thread_pool_working(std::vector<bool> &thread_state) {
    int count = 0;
    for (auto && i : thread_state) {
        if (i)
            count++;
    }
    if (count == 0)
        return false;
    return true;
}


arma::sp_mat sdp_branch_and_bound(int k, arma::mat &Ws, UserConstraints &constraints) {

    int n_thread = branch_and_bound_parallel;

    JobAbstractQueue *queue = nullptr;
    switch (branch_and_bound_visiting_strategy) {
        case DEPTH_FIRST:
            queue = new JobStack();
            break;
        case BEST_FIRST:
            queue = new JobPriorityQueue();
            break;
        case BREADTH_FIRST:
            queue = new JobQueue();
            break;
        default:
            queue = nullptr;
    }

    auto *shared_data = new SharedData();
    shared_data->global_ub = std::numeric_limits<double>::infinity();
    shared_data->n_nodes = 0;
    shared_data->sum_ineq = 0.0;
    shared_data->sum_cp_iter = 0.0;
    shared_data->queue = queue;

    shared_data->threadStates.reserve(n_thread);
    for (int i = 0; i < n_thread; i++) {
        shared_data->threadStates.push_back(false);
    }
    
    arma::mat C = Ws * Ws.t();
    double C_trace = arma::trace(C);

    auto *input_data = new InputData();
    input_data->Ws = Ws;
    input_data->C_trace = C_trace;
    input_data->k = k;
	//input_data->distances = compute_distances(Ws);

    ThreadPool pool(shared_data, input_data, n_thread);
    
    print_header_sdp(log_file);

    auto start_all = std::chrono::high_resolution_clock::now();
    
    auto *matlab_struct = new MatlabStruct();
    matlab_struct->matlabPtr = start_matlab(sdp_solver_folder);

    std::pair<JobData *, JobData *> jobs = build_root_problem(matlab_struct, input_data, shared_data, constraints);

    delete (matlab_struct);
    
    double root_gap = shared_data->gap;

    JobData *cl_job = jobs.first;
    JobData *ml_job = jobs.second;
    if (cl_job != nullptr && ml_job != nullptr) {
        pool.addJob(cl_job);
        pool.addJob(ml_job);
    }

    while (true) {

        {
            std::unique_lock<std::mutex> l(shared_data->queueMutex);
            while (is_thread_pool_working(shared_data->threadStates) && shared_data->n_nodes < branch_and_bound_max_nodes) {
                shared_data->mainConditionVariable.wait(l);
            }

            if (shared_data->queue->empty() || shared_data->n_nodes >= branch_and_bound_max_nodes)
                break;
        }

    }

    auto end_all = std::chrono::high_resolution_clock::now();
    auto duration_all = std::chrono::duration_cast<std::chrono::seconds>(end_all - start_all);

    pool.quitPool();

    if (queue->empty())
        shared_data->gap = 0.0;

    log_file << "\n";
    log_file << "WALL_TIME: " << duration_all.count() << " sec\n";
    log_file << "N_NODES: " << shared_data->n_nodes << "\n";
    log_file << "AVG_INEQ: " << (double) shared_data->sum_ineq / shared_data->n_nodes << "\n";
    log_file << "AVG_CP_ITER: " << (double) shared_data->sum_cp_iter / shared_data->n_nodes << "\n";
    log_file << "ROOT_GAP: " << std::max(0.0, root_gap) << "\n";
    log_file << "GAP: " << std::max(0.0, shared_data->gap) << "\n";
    log_file << "BEST: " << shared_data->global_ub << "\n\n";

    arma::sp_mat result = shared_data->global_X;
	save_X_to_file(result);

    // free memory

    delete (input_data);
    delete (queue);
    delete (shared_data);

    return result;

}
