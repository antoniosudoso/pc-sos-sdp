#include "ilp_model.h"

template<class T>
Matrix<T>::Matrix(int row, int col): rows(row), cols(col), data(rows*cols) {}

template<class T>
T &Matrix<T>::operator()(size_t row, size_t col) {
    return data[row*cols+col];
}

template<class T>
T Matrix<T>::operator()(size_t row, size_t col) const {
    return data[row*cols+col];
}

std::string ILP_model::get_variable_name(int i, int j){
	std::ostringstream os;
	os << i << " " << j;
	return os.str();
}
std::string ILP_model::get_row_sum_constraint_name(int i){
	std::ostringstream os;
	os << "row_sum " << i;
	return os.str();
}
std::string ILP_model::get_col_sum_constraint_name(int i){
	std::ostringstream os;
	os << "col_sum " << i;
	return os.str();
}
std::string ILP_model::get_cl_constraint_name(int i, int j, int h){
	std::ostringstream os;
	os << "cl " << i << " " << j << " " << h;
	return os.str();
}
std::string ILP_model::get_ml_constraint_name(int i, int j, int h){
	std::ostringstream os;
	os << "ml " << i << " " << j << " " << h;
	return os.str();
}

void ILP_model::compute_objective_function_constant(arma::mat &Ws, arma::vec &cardinalities, double C_trace){
	obj_function_constant = C_trace;
	for (int i = 0; i < n; i++){
		arma::vec point = Ws.row(i).t()/cardinalities(i);
		obj_function_constant -= arma::dot(point, point) * cardinalities(i);
	}
}





ILP_gurobi_model::ILP_gurobi_model(GRBEnv *env, int n, int k) : model(*env), X(n,k) {
	this->n = n;
	this->k = k;
	this->env = env;
	this->X = create_X_variables(this->model);
    this->model.set("OutputFlag", "0");
    this->model.set("Threads", "1");
    //this->model.set("Presolve", std::to_string(lp_solver_presolve));
}

Matrix<GRBVar> ILP_gurobi_model::create_X_variables(GRBModel &model) {
    Matrix<GRBVar> X(n, k);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
			std::string name = get_variable_name(i, j);
            X(i, j) = model.addVar(0.0, 1, 0.0, GRB_BINARY, name);
        }
    }
    return X;
}

arma::mat ILP_gurobi_model::get_solution() {
	arma::mat Xopt(n, k);
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < k; j++) {
			Xopt(i, j) = X(i, j).get(GRB_DoubleAttr_X);
		}
	}
	return Xopt;
}

int ILP_gurobi_model::get_n_constraints(){
	model.update();
	return model.get(GRB_IntAttr_NumConstrs);
}

double ILP_gurobi_model::get_value(){
	if (status == GRB_INFEASIBLE)
		return std::numeric_limits<double>::infinity();
	else
		return model.get(GRB_DoubleAttr_ObjVal);
}


void ILP_gurobi_model::set_objective_function(arma::mat &Ws, arma::vec &cardinalities, arma::mat &centroids) {
    GRBLinExpr obj = 0;
	double coefficient;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
			arma::vec diff = Ws.row(i).t()/cardinalities(i) - centroids.row(j).t();
			coefficient = arma::dot(diff, diff) * cardinalities(i);
            obj += coefficient * X(i, j);
        }
    }
	obj += obj_function_constant;
    model.setObjective(obj, GRB_MINIMIZE);
}

void ILP_gurobi_model::add_cannot_link_constraints(std::vector<std::pair<int, int>> &cl_pairs) {
    for (auto &elem : cl_pairs) {
        int i = elem.first;
        int j = elem.second;
		for (int h = 0; h < k; h++){
			std::string name = get_cl_constraint_name(i,j,h);
			model.addConstr(X(i, h) + X(j, h) <= 1, name);
		}
    }
}

void ILP_gurobi_model::add_must_link_constraints(std::vector<std::pair<int, int>> &ml_pairs) {
    for (auto &elem : ml_pairs) {
        int i = elem.first;
        int j = elem.second;
		for (int h = 0; h < k; h++){
			std::string name = get_ml_constraint_name(i,j,h);
			model.addConstr(X(i, h) - X(j, h) == 0, name);
		}
    }
}

void ILP_gurobi_model::add_row_sum_constraints() {
	GRBLinExpr lhs_sum = 0;
	for (int i = 0; i < n; i++) {
		lhs_sum = 0;
		for (int j = 0; j < k; j++) {
			lhs_sum += X(i, j);
		}
		// Xe_l = e
		std::string name = get_row_sum_constraint_name(i);
		model.addConstr(lhs_sum == 1, name);
	}
}

void ILP_gurobi_model::add_col_sum_constraints() {
	GRBLinExpr lhs_sum = 0;
	for (int j = 0; j < k; j++) {
		lhs_sum = 0;
		for (int i = 0; i < n; i++) {
			lhs_sum += X(i, j);
		}
		// X'e_l >= e
		std::string name = get_col_sum_constraint_name(j);
		model.addConstr(lhs_sum >= 1, name);
	}
}

void ILP_gurobi_model::optimize(){
	try {
		model.optimize();
		status = model.get(GRB_IntAttr_Status);
    } catch (GRBException &e) {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
}


