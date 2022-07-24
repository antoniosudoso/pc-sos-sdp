## PC-SOS-SDP: an Exact Solver for Semi-supervised Minimum Sum-of-Squares Clustering

<p align="center">
  <img src="https://github.com/antoniosudoso/pc-sos-sdp/blob/main/logo.svg" width="200" height="200" />
</p>


**PC-SOS-SDP** is an exact algorithm based on the branch-and-bound technique for solving the semi-supervised Minimum Sum-of-Squares Clustering (MSSC) problem with pairwise constraints (i.e. must-link and cannot-link constraints) described in the paper ["An Exact Algorithm for Semi-supervised Minimum Sum-of-Squares Clustering"](https://doi.org/10.1016/j.cor.2022.105958). This repository contains the C++ source code, the MATLAB scripts, and the datasets used for the experiments.

> V. Piccialli, A. Russo Russo, A. M. Sudoso, An Exact Algorithm for Semi-supervised Minimum Sum-of-Squares Clustering, 
> Computers & Operations Research 2022, https://doi.org/10.1016/j.cor.2022.105958.

## Installation
**PC-SOS-SDP** calls the semidefinite programming solver [SDPNAL+](https://blog.nus.edu.sg/mattohkc/softwares/sdpnalplus/) by using the [MATLAB Engine API](https://www.mathworks.com/help/matlab/calling-matlab-engine-from-cpp-programs.html) for C++. It requires the MATLAB engine library *libMatlabEngine* and the Matlab Data Array library *libMatlabDataArray*. **PC-SOS-SDP** calls the integer programming solver [Gurobi](https://www.gurobi.com/). **PC-SOS-SDP** uses the [Armadillo](http://arma.sourceforge.net/) library to handle matrices and linear algebra operations efficiently. Before installing Armadillo, first install OpenBLAS and LAPACK along with the corresponding development files. **PC-SOS-SDP** implements a configurable thread pool of POSIX threads to speed up the branch-and-bound search.

Ubuntu and Debian instructions:
1) Install MATLAB (>= 2016b)
2) Install Gurobi (>= 9.0)
3) Install CMake, OpenBLAS, LAPACK and Armadillo:
 ```
sudo apt-get update
sudo apt-get install cmake libopenblas-dev liblapack-dev libarmadillo-dev
```
4) Open the makefile `clustering_c++/Makefile` 
	- Set the variable `matlab_path` with your MATLAB folder.
	- Set the variable `gurobi_path` with your Gurobi folder.
5) Compile the code:

```
cd clustering_c++/
make
```

4) Download SDPNAL+, move the folder `clustering_matlab` containing the MATLAB source code of **PC-SOS-SDP** in the SDPNAL+ main directory and set the parameter `SDP_SOLVER_FOLDER` of the configuration file accordingly. This folder and its subfolders will be automatically added to the MATLAB search path when **PC-SOS-SDP** starts.

The code has been tested on Ubuntu Server 20.04 with MATLAB R2020b, Gurobi 9.2 and Armadillo 10.2.

## Configuration
Various parameters used in **PC-SOS-SDP** can be modified in the configuration file `clustering_c++/config.txt`:

- `BRANCH_AND_BOUND_TOL` - optimality tolerance of the branch-and-bound
- `BRANCH_AND_BOUND_PARALLEL` -  thread pool size: single thread (1), multi-thread (> 1)
- `BRANCH_AND_BOUND_MAX_NODES` - maximum number of nodes
- `BRANCH_AND_BOUND_VISITING_STRATEGY` - best first (0),  depth first (1), breadth first (2)
- `SDP_SOLVER_SESSION_THREADS_ROOT` - number of threads for the MATLAB session at the root
- `SDP_SOLVER_SESSION_THREADS` - number of threads for the MATLAB session for the ML and CL nodes
- `SDP_SOLVER_FOLDER` - full path of the SDPNAL+ folder
- `SDP_SOLVER_TOL` - accuracy of SDPNAL+
- `SDP_SOLVER_VERBOSE` - do not display log (0), display log (1)
- `SDP_SOLVER_MAX_CP_ITER_ROOT` - maximum number of cutting-plane iterations at the root
- `SDP_SOLVER_MAX_CP_ITER` - maximum number of cutting-plane iterations for the ML and CL nodes
- `SDP_SOLVER_CP_TOL` - cutting-plane tolerance between two consecutive cutting-plane iterations
- `SDP_SOLVER_MAX_INEQ` - maximum number of valid inequalities to add
- `SDP_SOLVER_INHERIT_PERC` - fraction of inequalities to inherit
- `SDP_SOLVER_EPS_INEQ` - tolerance for checking the violation of the inequalities
- `SDP_SOLVER_EPS_ACTIVE` - tolerance for detecting the active inequalities
- `SDP_SOLVER_MAX_PAIR_INEQ` - maximum number of pair inequalities to separate
- `SDP_SOLVER_PAIR_PERC` - fraction of the most violated pair inequalities to add
- `SDP_SOLVER_MAX_TRIANGLE_INEQ` - maximum number of triangle inequalities to separate
- `SDP_SOLVER_TRIANGLE_PERC` - fraction of the most violated triangle inequalities to add
 
## Usage
```
cd clustering_c++/
./bb <DATASET> <K> <CONSTRAINTS> <LOG> <RESULT>
```
- `DATASET` - path of the dataset
- `K` - number of clusters
- `CONSTRAINTS` - path of the constraints
- `LOG` - path of the log file
- `RESULT` - path of the optimal cluster assignment matrix

File `DATASET` contains the data points `x_ij` and the must include an header line with the problem size `n` and the dimension `d`:

```
n d
x_11 x_12 ... x_1d
x_21 x_22 ... x_2d
...
...
x_n1 x_n2 ... x_nd
```

File `CONSTRAINTS` should include indices `(i, j)` of the data points involved in must-link (ML) and/or cannot-link (CL) constraints:

```
CL i1 j1
CL i2 j2
...
...
ML i3 j3
ML i4 j4
```

If it does not contain any constraint (empty file), **PC-SOS-SDP** becomes [SOS-SDP](https://github.com/antoniosudoso/sos-sdp/) (the exact solver for unsupervised MSSC).

## Log

The log file reports the progress of the algorithm:

- `N` - size of the current node
- `NODE_PAR` - id of the parent node
- `NODE` - id of the current node
- `LB_PAR` - lower bound of the parent node
- `LB` - lower bound of the current node
- `FLAG` - termination flag of SDPNAL+
    -  `0` - SDP is solved to the required accuracy
    -  `1` - SDP is not solved successfully
    -  `-1, -2, -3` - SDP is partially solved successfully
- `TIME (s)` - running time in seconds of the current node
- `CP_ITER` - number of cutting-plane iterations
- `CP_FLAG` - termination flag of the cutting-plane procedure
    - `-3` - current bound is worse than the previous one
    - `-2` - SDP is not solved successfully
    - `-1` - maximum number of iterations
    -  `0` - no violated inequalities
    -  `1` - maximum number of inequalities
    -  `2` - node must be pruned
    -  `3` - cutting-plane tolerance
- `CP_INEQ` - number of inequalities added in the last cutting-plane iteration
- `PAIR TRIANGLE CLIQUE` - average number of added cuts for each class of inequalities
- `UB` - current upper bound
- `GUB` - global upper bound
- `I J` - current branching decision
- `NODE_GAP` - gap at the current node
- `GAP` - overall gap 
- `OPEN` - number of open nodes

Log file example:

```
DATA_PATH, n, d, k: /home/ubuntu/PC-SOS-SDP/instances/glass.txt 214 9 6
CONSTRAINTS_PATH: /home/ubuntu/PC-SOS-SDP/instances/constraints/glass/ml_50_cl_50_3.txt
LOG_PATH: /home/ubuntu/PC-SOS_SDP/logs/glass/log_ml_50_cl_50_3.txt

BRANCH_AND_BOUND_TOL: 1e-4
BRANCH_AND_BOUND_PARALLEL: 16
BRANCH_AND_BOUND_MAX_NODES: 200
BRANCH_AND_BOUND_VISITING_STRATEGY: 0

SDP_SOLVER_SESSION_THREADS_ROOT: 16
SDP_SOLVER_SESSION_THREADS: 1
SDP_SOLVER_FOLDER: /home/ubuntu/PC-SOS-SDP/SDPNAL+/
SDP_SOLVER_TOL: 1e-05
SDP_SOLVER_VERBOSE: 0
SDP_SOLVER_MAX_CP_ITER_ROOT: 80
SDP_SOLVER_MAX_CP_ITER: 40
SDP_SOLVER_CP_TOL: 1e-06
SDP_SOLVER_MAX_INEQ: 100000
SDP_SOLVER_INHERIT_PERC: 1
SDP_SOLVER_EPS_INEQ: 0.0001
SDP_SOLVER_EPS_ACTIVE: 1e-06
SDP_SOLVER_MAX_PAIR_INEQ: 100000
SDP_SOLVER_PAIR_PERC: 0.05
SDP_SOLVER_MAX_TRIANGLE_INEQ: 100000
SDP_SOLVER_TRIANGLE_PERC: 0.05


|    N| NODE_PAR|    NODE|      LB_PAR|          LB|  FLAG|  TIME (s)| CP_ITER| CP_FLAG|   CP_INEQ|     PAIR  TRIANGLE    CLIQUE|          UB|         GUB|     I      J|     NODE_GAP|          GAP|  OPEN|
|  164|       -1|       0|        -inf|     93.3876|     0|       110|       7|      -3|      6456|  242.571      4802   8.14286|     93.5225|    93.5225*|    -1     -1|   0.00144229|   0.00144229|     0|
|  163|        0|       1|     93.3876|     93.4388|     0|        35|       2|      -3|      5958|        1      3675         0|     93.4777|    93.4777*|    79    142|  0.000416211|  0.000416211|     0|
|  164|        0|       2|     93.3876|     93.4494|     0|        47|       2|      -3|      6888|        0      4635         0|     93.5225|     93.4777|    79    142|  0.000302427|  0.000302427|     0|
|  162|        1|       3|     93.4388|      93.506|     0|        27|       1|       2|      6258|        9      3759         0|         inf|     93.4777|   119    152| -0.000302724| -0.000302724|     0|
|  163|        1|       4|     93.4388|     93.4536|     0|        47|       4|      -3|      3336|        0      1789         0|     93.4777|     93.4777|   119    152|   0.00025747|   0.00025747|     0|
|  164|        2|       5|     93.4494|     93.4549|     0|        37|       1|      -3|      6888|        0      5000         0|     93.5225|     93.4777|    47     54|  0.000243844|  0.000243844|     0|
|  163|        2|       6|     93.4494|     93.4708|     0|        51|       2|       2|      7292|       11      4693         0|     93.5559|     93.4777|    47     54|  7.36443e-05|  7.36443e-05|     0|
|  164|        5|       7|     93.4549|      93.475|     0|        22|       0|       2|      6888|        0         0         0|     93.5225|     93.4777|   122    153|  2.82805e-05|  2.82805e-05|     0|
|  163|        4|       8|     93.4536|     93.4536|     0|        38|       2|      -3|      3257|        0     668.5         0|     93.4704|    93.4704*|    47     54|  0.000180057|  0.000180057|     0|
|  163|        5|       9|     93.4549|     93.5216|     0|        41|       1|       2|      6893|        8      5000         0|         inf|     93.4704|   122    153| -0.000547847| -0.000547847|     0|
|  163|        8|      10|     93.4536|     93.4536|     0|        27|       1|      -3|      3257|        0       879         0|     93.4704|     93.4704|    37     45|  0.000180057|  0.000180057|     0|
|  162|        8|      11|     93.4536|     93.4838|     0|        33|       1|       2|      6158|       24      4233         0|         inf|     93.4704|    37     45| -0.000143677| -0.000143677|     0|
|  162|        4|      12|     93.4536|     93.4658|     0|        75|       5|      -3|      2793|      4.6      2379         0|     93.5111|     93.4704|    47     54|  4.89954e-05|  4.89954e-05|     0|
|  162|       10|      13|     93.4536|     93.5053|     0|        19|       0|       2|      3122|        0         0         0|         inf|     93.4704|    37     99|  -0.00037365|  -0.00037365|     0|
|  163|       10|      14|     93.4536|     93.4701|     0|        31|       0|       2|      3257|        0         0         0|     93.4704|     93.4704|    37     99|  3.13989e-06|  3.13989e-06|     0|

WALL_TIME: 304 sec
N_NODES: 15
AVG_INEQ: 2788.05
AVG_CP_ITER: 1.93333
ROOT_GAP: 0.00144229
GAP: 0
BEST: 93.4704
```
