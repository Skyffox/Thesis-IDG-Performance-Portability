#include <omp.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>

#include "init.h"
#include "gridder.h"

using namespace std::chrono;


int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: ./program kernel_iterations\n");
        exit(0);
    }

    auto begin_create = steady_clock::now();
    idg::Array2D<idg::UVWCoordinate<float>> uvw(NR_BASELINES, NR_TIMESTEPS);
    idg::Array1D<float> frequencies(NR_CHANNELS);
    idg::Array1D<float> wavenumbers(NR_CHANNELS);
    idg::Array3D<idg::Visibility<std::complex<float>>> visibilities(NR_BASELINES, NR_TIMESTEPS, NR_CHANNELS);
    idg::Array1D<idg::Baseline> baselines(NR_BASELINES);
    idg::Array2D<float> spheroidal(SUBGRID_SIZE, SUBGRID_SIZE);
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms(NR_TIMESLOTS, NR_STATIONS, SUBGRID_SIZE, SUBGRID_SIZE);
    idg::Array1D<idg::Metadata> metadata(NR_SUBGRIDS);
    idg::Array4D<std::complex<float>> subgrids(NR_SUBGRIDS, NR_CORRELATIONS, SUBGRID_SIZE, SUBGRID_SIZE);
    auto end_create = steady_clock::now();

    // Initialize random number generator
    srand(0);

    // Initialize data structures
    auto begin_init = steady_clock::now();
    initialize_uvw(GRID_SIZE, uvw);
    initialize_frequencies(frequencies);
    initialize_wavenumbers(frequencies, wavenumbers);
    initialize_visibilities(GRID_SIZE, IMAGE_SIZE, frequencies, uvw, visibilities);
    initialize_baselines(NR_STATIONS, baselines);
    initialize_spheroidal(spheroidal);
    initialize_aterms(spheroidal, aterms);
    initialize_metadata(GRID_SIZE, NR_TIMESLOTS, NR_TIMESTEPS_SUBGRID, baselines, metadata);
    initialize_subgrids(subgrids);
    auto end_init = steady_clock::now();

    // WARMUP
    for (int i = 0; i < 5; i++) {
        kernel_gridder(
            NR_SUBGRIDS, GRID_SIZE, SUBGRID_SIZE, IMAGE_SIZE, W_STEP, NR_CHANNELS, NR_STATIONS,
            uvw.data(), wavenumbers.data(), (std::complex<float> *) visibilities.data(),
            (float *) spheroidal.data(), (std::complex<float> *) aterms.data(), metadata.data(),
            subgrids.data());
    }

    // Run reference
    auto begin_kernel = steady_clock::now();
    for (int i = 0; i < atoi(argv[1]); i++) {
        kernel_gridder(
            NR_SUBGRIDS, GRID_SIZE, SUBGRID_SIZE, IMAGE_SIZE, W_STEP, NR_CHANNELS, NR_STATIONS,
            uvw.data(), wavenumbers.data(), (std::complex<float> *) visibilities.data(),
            (float *) spheroidal.data(), (std::complex<float> *) aterms.data(), metadata.data(),
            subgrids.data());
    }
    auto end_kernel = steady_clock::now();

    auto create_time = duration_cast<nanoseconds>(end_create - begin_create).count();
    auto init_time = duration_cast<nanoseconds>(end_init - begin_init).count();
    auto kernel_time = duration_cast<nanoseconds>(end_kernel - begin_kernel).count();

    std::cout << ">>> Kernel iterations: " << atoi(argv[1]) << std::endl;
    std::cout << ">> Object creation: " << create_time << std::endl;
    std::cout << ">> Object initialisation: " << init_time << std::endl;
    std::cout << ">> Kernel duration: " << kernel_time << std::endl;
    std::cout << std::endl;

    return EXIT_SUCCESS;
}