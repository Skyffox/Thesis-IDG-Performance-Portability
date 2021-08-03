# Compiler flags
CXX = icpx

# Intel I5 personal machine
CXXFLAGS_CPU = -std=c++17 -march=native -O3 -fiopenmp

# iris optimization flags
CXXFLAGS_GPU = -std=c++17 -march=native -O3 -fiopenmp -fopenmp-targets=spir64


all:: gridder-cpu gridder-gpu

gridder-cpu:
	$(CXX) $(CXXFLAGS_CPU) -o run-gridder-cpu gridder.cpp init.cpp run-gridder.cpp

gridder-gpu:
	$(CXX) $(CXXFLAGS_GPU) -o run-gridder-gpu gridder.cpp init.cpp run-gridder.cpp

clean:
	rm -rf run-gridder-cpu run-gridder-gpu
