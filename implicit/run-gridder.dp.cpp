// OPNEAPI LIBS
#include <CL/sycl.hpp>

// EXTRA LIBS
#include <chrono>

// THEIR LIBS
#include "init.h"
#include "gridder.h"
#include "print.h"

// Some calls also exist in the std library so we only use one namespace to avoid ambiguity
using namespace sycl;
using namespace std::chrono;


void output_dev_info(const device& dev) {
    std::cout << ">> Selected device: " << dev.get_info<info::device::name>() << "\n";
    // std::cout << ">> Max compute units: " << dev.get_info<info::device::max_compute_units>() << "\n";
}


int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: ./program kernel_iterations\n");
        exit(0);
    }

    queue q( default_selector{} );
    output_dev_info(device{ default_selector{} });

    auto begin_create = steady_clock::now();
    float *u = (float *) malloc_shared(NR_BASELINES * NR_TIMESTEPS * sizeof(float), q);
    float *v = (float *) malloc_shared(NR_BASELINES * NR_TIMESTEPS * sizeof(float), q);
    float *w = (float *) malloc_shared(NR_BASELINES * NR_TIMESTEPS * sizeof(float), q);
    double *frequencies = (double *) malloc_shared(NR_CHANNELS * sizeof(double), q);
    float *wavenumbers = (float *) malloc_shared(NR_CHANNELS * sizeof(float), q);
    std::array<std::complex<float>, 4> *visibilities = (std::array<std::complex<float>, 4> *) malloc_shared(NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS * sizeof(float) * 8, q);
    std::array<int, 2> *stations = (std::array<int, 2> *) malloc_shared(NR_BASELINES * sizeof(int) * 2, q);
    float *spheroidal = (float *) malloc_shared(SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float), q);
    std::array<std::complex<float>, 4> *aterms = (std::array<std::complex<float>, 4> *) malloc_shared(NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float) * 8, q);
    std::array<int, 9> *metadata = (std::array<int, 9> *) malloc_shared(NR_BASELINES * NR_TIMESLOTS * sizeof(int) * 9, q);
    std::complex<float> *subgrid = (std::complex<float> *) malloc_shared(NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float) * 2, q);
    auto end_create = steady_clock::now();

    // Initialize random number generator.
    srand(0);

    auto begin_init = steady_clock::now();
    initialize_uvw(u, v, w);
    initialize_frequencies(frequencies);
    initialize_wavenumbers(frequencies, wavenumbers);
    initialize_visibilities(frequencies, u, v, visibilities);
    initialize_baselines(stations);
    initialize_spheroidal(spheroidal);
    initialize_aterms(spheroidal, aterms);
    initialize_metadata(stations, metadata);
    initialize_subgrids(subgrid);
    auto end_init = steady_clock::now();

    // printUVW(u, v, w);               // NOTE: Prints a lot
    // printFrequencies(frequencies);
    // printWavenumbers(wavenumbers);
    // printVisibilities(visibilities); // NOTE: Prints a lot
    // printBaselines(stations);
    // printSpheroidal(spheroidal);
    // printAterms(aterms);             // NOTE: Prints a lot
    // printMetadata(metadata);
    // printSubgrid(subgrid);          // NOTE: Prints a lot

    // WARMUP
    for (int i = 0; i < 5; i++) {
        kernel_gridder(
            q, NR_SUBGRIDS, GRID_SIZE, SUBGRID_SIZE, IMAGE_SIZE, W_STEP, NR_CHANNELS, NR_STATIONS,
            u, v, w, wavenumbers, visibilities, spheroidal, aterms, metadata, subgrid
        );
    }

    // Run gridder
    auto begin_kernel = steady_clock::now();
    for (int i = 0; i < atoi(argv[1]); i++) {
        kernel_gridder(
            q, NR_SUBGRIDS, GRID_SIZE, SUBGRID_SIZE, IMAGE_SIZE, W_STEP, NR_CHANNELS, NR_STATIONS,
            u, v, w, wavenumbers, visibilities, spheroidal, aterms, metadata, subgrid
        );
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
