#ifndef CLUSTERING_ILP_MODEL_H
#define CLUSTERING_ILP_MODEL_H

#include <armadillo>
#include <gurobi_c++.h>
#include <list>
#include "config_params.h"

template<class T>
class Matrix {

public: int rows;
public: int cols;
private: std::vector<T> data;

public:

    Matrix(int row, int col);

    T & operator()(size_t row, size_t col);
    T operator()(size_t row, size_t col) const;

};


class ILP_model {

	protected:
	int status;
	int n, k;
	double obj_function_constant;

	std::string get_variable_name(int i, int j);
	std::string get_row_sum_constraint_name(int i);
	std::string get_col_sum_constraint_name(int i);
	std::string get_trace_constraint_name();
	std::string get_cl_constraint_name(int i, int j, int h);
	std::string get_ml_constraint_name(int i, int j, int h);
	
	public:
	virtual void compute_objective_function_constant(arma::mat &Ws, arma::vec &cardinalities, double C_trace);
	virtual int get_n_constraints() = 0;
	virtual void set_objective_function(arma::mat &Ws, arma::vec &cardinalities, arma::mat &centroids) = 0;
	virtual void add_cannot_link_constraints(std::vector<std::pair<int, int>> &cl_pairs) = 0; 
	virtual void add_must_link_constraints(std::vector<std::pair<int, int>> &ml_pairs) = 0; 
	virtual void add_row_sum_constraints() = 0;
	virtual void add_col_sum_constraints() = 0;
	virtual void optimize() = 0;
	virtual double get_value() = 0;
	virtual arma::mat get_solution() = 0;
};

class ILP_gurobi_model : public ILP_model {

private:
	GRBEnv *env;
	GRBModel model;
	Matrix<GRBVar> X;

	Matrix<GRBVar> create_X_variables(GRBModel &model);


public:
	ILP_gurobi_model(GRBEnv *env, int n, int k);
	virtual int get_n_constraints();
	virtual void set_objective_function(arma::mat &Ws, arma::vec &cardinalities, arma::mat &centroids);
	virtual void add_cannot_link_constraints(std::vector<std::pair<int, int>> &cl_pairs); 
	virtual void add_must_link_constraints(std::vector<std::pair<int, int>> &ml_pairs); 
	virtual void add_row_sum_constraints();
	virtual void add_col_sum_constraints();
	virtual void optimize();
	virtual double get_value();
	virtual arma::mat get_solution();
};

#endif //CLUSTERING_ILP_MODEL_H
