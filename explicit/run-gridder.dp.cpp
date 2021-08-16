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
}


int main(int argc, char **argv)
{
    queue q( default_selector{} );
    output_dev_info(device{ default_selector{} });

    // NOTE host arrays
    auto begin_create_host = steady_clock::now();
    float* u = new float[NR_BASELINES * NR_TIMESTEPS];
    float* v = new float[NR_BASELINES * NR_TIMESTEPS];
    float* w = new float[NR_BASELINES * NR_TIMESTEPS];
    double* frequencies = new double[NR_CHANNELS];
    float* wavenumbers = new float[NR_CHANNELS];
    std::array<int, 2>* stations = new std::array<int, 2>[NR_BASELINES];
    float* spheroidal = new float[SUBGRID_SIZE * SUBGRID_SIZE];
    std::array<std::complex<float>, 4>* visibilities = new std::array<std::complex<float>, 4>[NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS];
    std::array<std::complex<float>, 4>* aterms = new std::array<std::complex<float>, 4>[NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE];
    std::array<int, 9>* metadata = new std::array<int, 9>[NR_BASELINES * NR_TIMESLOTS];
    std::complex<float>* subgrid = new std::complex<float>[NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE];
    auto end_create_host = steady_clock::now();

    // NOTE alloc memory on device
    auto begin_create_device = steady_clock::now();
    auto device_u = malloc_device<float>(NR_BASELINES * NR_TIMESTEPS, q);
    q.memset(device_u, 0, NR_BASELINES * NR_TIMESTEPS * sizeof(float)).wait();

    auto device_v = malloc_device<float>(NR_BASELINES * NR_TIMESTEPS, q);
    q.memset(device_v, 0, NR_BASELINES * NR_TIMESTEPS * sizeof(float)).wait();

    auto device_w = malloc_device<float>(NR_BASELINES * NR_TIMESTEPS, q);
    q.memset(device_w, 0, NR_BASELINES * NR_TIMESTEPS * sizeof(float)).wait();

    auto device_wavenumbers = malloc_device<float>(NR_CHANNELS, q);
    q.memset(device_wavenumbers, 0, NR_CHANNELS * sizeof(float)).wait();

    auto device_visibilities = malloc_device<std::array<std::complex<float>, 4>>(NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS, q);
    q.memset(device_visibilities, 0, NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS * sizeof(float) * 8).wait();

    auto device_spheroidal = malloc_device<float>(SUBGRID_SIZE * SUBGRID_SIZE, q);
    q.memset(device_spheroidal, 0, SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float)).wait();

    auto device_aterms = malloc_device<std::array<std::complex<float>, 4>>(NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE, q);
    q.memset(device_aterms, 0, NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float) * 8).wait();

    auto device_metadata = malloc_device<std::array<int, 9>>(NR_BASELINES * NR_TIMESLOTS, q);
    q.memset(device_metadata, 0, NR_BASELINES * NR_TIMESLOTS * sizeof(int) * 9).wait();

    auto device_subgrid = malloc_device<std::complex<float>>(NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE, q);
    q.memset(device_subgrid, 0, NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float) * 2).wait();
    auto end_create_device = steady_clock::now();

    // Initialize random number generator.
    srand(0);

    // std::cout << ">>> Initialize data structures on host" << std::endl;
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

    // copy from host to device
    auto begin_copy_to_device = steady_clock::now();
    q.memcpy(device_u, u, NR_BASELINES * NR_TIMESTEPS * sizeof(float)).wait();
    q.memcpy(device_v, v, NR_BASELINES * NR_TIMESTEPS * sizeof(float)).wait();
    q.memcpy(device_w, w, NR_BASELINES * NR_TIMESTEPS * sizeof(float)).wait();
    q.memcpy(device_wavenumbers, wavenumbers, NR_CHANNELS * sizeof(float)).wait();
    q.memcpy(device_spheroidal, spheroidal, SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float)).wait();
    q.memcpy(device_visibilities, visibilities, NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS * sizeof(float) * 8).wait();
    q.memcpy(device_aterms, aterms, NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float) * 8).wait();
    q.memcpy(device_metadata, metadata, NR_BASELINES * NR_TIMESLOTS * sizeof(int) * 9).wait();
    q.memcpy(device_subgrid, subgrid, NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float) * 2).wait();
    auto end_copy_to_device = steady_clock::now();

    // WARMUP
    kernel_gridder(
        q, NR_SUBGRIDS, GRID_SIZE, SUBGRID_SIZE, IMAGE_SIZE, W_STEP, NR_CHANNELS, NR_STATIONS,
        device_u, device_v, device_w, device_wavenumbers, visibilities, device_spheroidal,
        aterms, device_metadata, subgrid
    );

    // Run reference
    auto begin_kernel = steady_clock::now();
    kernel_gridder(
        q, NR_SUBGRIDS, GRID_SIZE, SUBGRID_SIZE, IMAGE_SIZE, W_STEP, NR_CHANNELS, NR_STATIONS,
        device_u, device_v, device_w, device_wavenumbers, device_visibilities, device_spheroidal,
        device_aterms, device_metadata, device_subgrid
    );
    auto end_kernel = steady_clock::now();

    auto begin_copy_to_host = steady_clock::now();
    q.memcpy(subgrid, device_subgrid, NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float) * 2).wait();
    auto end_copy_to_host = steady_clock::now();

    auto create_time_host = duration_cast<nanoseconds>(end_create_host - begin_create_host).count();
    auto create_time_device = duration_cast<nanoseconds>(end_create_device - begin_create_device).count();
    auto init_time = duration_cast<nanoseconds>(end_init - begin_init).count();
    auto copy_to_device = duration_cast<nanoseconds>(end_copy_to_device - begin_copy_to_device).count();
    auto kernel_time = duration_cast<nanoseconds>(end_kernel - begin_kernel).count();
    auto copy_to_host = duration_cast<nanoseconds>(end_copy_to_host - begin_copy_to_host).count();

    std::cout << ">> Object creation (host): " << create_time_host << std::endl;
    std::cout << ">> Object creation (device): " << create_time_device << std::endl;
    std::cout << ">> Object initialisation: " << init_time << std::endl;
    std::cout << ">> H2D: " << copy_to_device << std::endl;
    std::cout << ">> Kernel duration: " << kernel_time << std::endl;
    std::cout << ">> D2H: " << copy_to_host << std::endl;
    std::cout << std::endl;

    // NOTE only free the device arrays
    free(device_u, q);
    free(device_v, q);
    free(device_w, q);
    free(device_wavenumbers, q);
    free(device_spheroidal, q);
    free(device_visibilities, q);
    free(device_aterms, q);
    free(device_metadata, q);
    free(device_subgrid, q);

    return EXIT_SUCCESS;
}
