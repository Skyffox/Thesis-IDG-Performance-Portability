# Compiler flags
CXX = dpcpp

CXXFLAGS = -w -g -std=c++17 -O2 -I..

# Run the compiler with more optimization flags
# CXXFLAGS = -w -g -std=c++17 -march=native -O3 -I..

# SYCL flags
SYCL_LDFLAGS = -fsycl

CXXSOURCES = init.dp.cpp gridder.dp.cpp run-gridder.dp.cpp
OBJECT_FILES = $(CXXSOURCES:%.dp.cpp=%.o)
EXECUTABLES = $(CXXSOURCES:%.dp.cpp=%.x)

default:: run-gridder.x

run-%.x: run-%.o init.o %.o
	${CXX} ${SYCL_LDFLAGS} -o $@ $^

%.o: %.dp.cpp
	${CXX} -c ${CXXFLAGS} ${SYCL_LDFLAGS} -o $@ $<

clean::
	rm -rf $(EXECUTABLES) $(OBJECT_FILES)
