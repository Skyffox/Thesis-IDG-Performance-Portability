#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <array>

using namespace std::chrono;


void VectorAdd(int N, std::array<int, 10000> a, std::array<int, 10000> &b, std::array<int, 10000> &c)
{
    // NOTE: WORKS
    #pragma omp target map(tofrom: c), map(to: a, b, N)
    #pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        c[i] = a[i] + b[i];
    }
}


int main(int argc, char const *argv[]) {
    const int array_size = 10000;

    std::array<int, array_size> a;
    std::array<int, array_size> b;
    std::array<int, array_size> c;
    std::array<int, array_size> c_serial;

    for (size_t i = 0; i < array_size; i++) {
        a[i] = i;
        b[i] = i;
    }

    // Vector addition in DPC++.
    auto begin_kernel = steady_clock::now();
    for (int i = 0; i < 2; i++) {
        VectorAdd(array_size, a, b, c);
    }
    auto end_kernel = steady_clock::now();

    auto kernel_time = duration_cast<nanoseconds>(end_kernel - begin_kernel).count();
    std::cout << kernel_time << std::endl;

    for (int i =0; i < array_size; i++) {
        c_serial[i] = a[i] + b[i];
    }

    for (int i = 0; i < array_size; i++) {
    	if (c_serial[i] != c[i]) {
    	    std::cout << "vector add failed on device" << std::endl;
            std::cout << c_serial[i] << " " << c[i] << std::endl;
    	}
    }

    return 0;
}
