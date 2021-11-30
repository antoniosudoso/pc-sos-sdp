matlab_path = /usr/local/MATLAB/R2020a
gurobi_path = /opt/gurobi902/linux64

sources = main.cpp matlab_util.cpp sdp_solver_util.cpp sdp_branch_and_bound.cpp kmeans_util.cpp Kmeans.cpp JobQueue.cpp util.cpp ThreadPool.cpp feasibility.cpp ub_heuristics.cpp ilp_model.cpp
objects = $(sources:.cpp=.o)

all: bb

bb: $(objects)
	g++ -L $(matlab_path)/extern/bin/glnxa64/ -L $(gurobi_path)/lib/ -o $@ $^ -larmadillo -pthread -llapack -lopenblas -lMatlabDataArray -lMatlabEngine -lgurobi_c++ -lgurobi90 -Wl,-rpath,$(matlab_path)/extern/bin/glnxa64/ -Wl,-rpath,$(gurobi_path)/lib

%.o: %.cpp
		g++ -O2 -o $@ -c $< -I $(matlab_path)/extern/include/ -I $(gurobi_path)/include


.PHONY: clean
		
clean:
		rm -f bb *.o
