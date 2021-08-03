# Compiler flags
CXX = dpcpp

# Run the compiler with more optimization flags
CXXFLAGS = -std=c++17 -march=native -O3

all: compare

compare:
	$(CXX) $(CXXFLAGS) -o run-compare compare.cpp
