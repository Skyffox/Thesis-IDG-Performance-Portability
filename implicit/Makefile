# Compiler
CXX = dpcpp

# Optimization flags
CXXFLAGS = -std=c++17 -march=native -O3 -fsycl

all:: gridder

gridder:
	$(CXX) $(CXXFLAGS) -o run-gridder gridder.dp.cpp init.dp.cpp run-gridder.dp.cpp

clean:
	rm -rf run-gridder
