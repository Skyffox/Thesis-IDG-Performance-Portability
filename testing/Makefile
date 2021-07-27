# Compiler flags
CXX = icpx

# Intel I5 personal machine (NOTE: maybe add back -fsycl for all)
CXXFLAGS = -std=c++17 -march=native -O3 -fiopenmp

# iris optimization flags
#CXXFLAGS = -std=c++17 -march=native -O3 -fiopenmp -fopenmp-targets=spir64

# gen9 optimization flags (TEST WHETHER -march=native is applicable)
#CXXFLAGS = -std=c++17 -O3 -fiopenmp -fopenmp-targets=spir64


all:: test

test:
	$(CXX) $(CXXFLAGS) -o run-test ref.cpp

clean:
	rm -rf run-test
