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


class CustomDeviceSelector : public device_selector {
    public:
        int operator()(const device &dev) const override {
            int device_rating = 0;
            // In the below code we are querying for the custom device. We query for a
            if (dev.is_gpu()) {
                device_rating = 2;
            } else if (dev.is_cpu()) {
                device_rating = 1;
            }
            return device_rating;
        };
};


void output_dev_info(const device& dev, const std::string& selector_name) {
    std::cout << ">> Selected device: " << dev.get_info<info::device::name>() << "\n";
    std::cout << ">> The Device Max Work Group Size is: " << dev.get_info<info::device::max_work_group_size>() << "\n";
    std::cout << ">> The Device Max Compute Units is: " << dev.get_info<info::device::max_compute_units>() << "\n";
}


int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: ./program kernel_iterations\n");
        exit(0);
    }

    // Create queue to push work to an accelerator
    // queue q( cpu_selector{} );

    // NOTE: if we want to run on a custom device
    queue q( CustomDeviceSelector{} );

    // Extra info of the device
    output_dev_info(device{CustomDeviceSelector{}}, "custom_selector" );

    // NOTE host arrays
    auto begin_create_host = high_resolution_clock::now();
    float u[NR_BASELINES * NR_TIMESTEPS];
    float v[NR_BASELINES * NR_TIMESTEPS];
    float w[NR_BASELINES * NR_TIMESTEPS];
    double frequencies[NR_CHANNELS];
    float wavenumbers[NR_CHANNELS];
    std::array<int, 2> stations[NR_BASELINES];
    float spheroidal[SUBGRID_SIZE * SUBGRID_SIZE];
    // std::array<std::complex<float>, 4> aterms[NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE];
    std::array<int, 9> metadata[NR_BASELINES * NR_TIMESLOTS];
    // std::array<std::complex<float>, NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE> subgrid;
    auto end_create_host = high_resolution_clock::now();

    // NOTE device arrays
    auto begin_create_device = high_resolution_clock::now();
    float *device_u = (float *) malloc_device(NR_BASELINES * NR_TIMESTEPS * sizeof(float), q);
    float *device_v = (float *) malloc_device(NR_BASELINES * NR_TIMESTEPS * sizeof(float), q);
    float *device_w = (float *) malloc_device(NR_BASELINES * NR_TIMESTEPS * sizeof(float), q);
    float *device_wavenumbers = (float *) malloc_device(NR_CHANNELS * sizeof(float), q);

    // NOTE: The visiblities and subgrid arrays are simply too big to copy so I will just use a shared array to make it easier for myself
    std::array<std::complex<float>, 4> *visibilities = (std::array<std::complex<float>, 4> *) malloc_shared(NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS * sizeof(float) * 8, q);
    std::complex<float> *subgrid = (std::complex<float> *) malloc_shared(NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float) * 2, q);
    std::array<std::complex<float>, 4> *aterms = (std::array<std::complex<float>, 4> *) malloc_shared(NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float) * 8, q);

    float *device_spheroidal = (float *) malloc_device(SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float), q);
    // std::array<std::complex<float>, 4> *device_aterms = (std::array<std::complex<float>, 4> *) malloc_device(NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float) * 8, q);
    std::array<int, 9> *device_metadata = (std::array<int, 9> *) malloc_device(NR_BASELINES * NR_TIMESLOTS * sizeof(int) * 9, q);
    // std::complex<float> *device_subgrid = (std::complex<float> *) malloc_device(NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float) * 2, q);
    auto end_create_device = high_resolution_clock::now();

    // Initialize random number generator.
    srand(0);

    // std::cout << ">>> Initialize data structures on host" << std::endl;
    auto begin_init = high_resolution_clock::now();
    initialize_uvw(u, v, w);
    initialize_frequencies(frequencies);
    initialize_wavenumbers(frequencies, wavenumbers);
    initialize_visibilities(frequencies, u, v, visibilities);
    initialize_baselines(stations);
    initialize_spheroidal(spheroidal);
    initialize_aterms(spheroidal, aterms);
    initialize_metadata(stations, metadata);
    initialize_subgrids(subgrid);
    auto end_init = high_resolution_clock::now();

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
    auto begin_copy_to_device = high_resolution_clock::now();
    q.memcpy(device_u, &u, NR_BASELINES * NR_TIMESTEPS * sizeof(float));
    q.memcpy(device_v, &v, NR_BASELINES * NR_TIMESTEPS * sizeof(float));
    q.memcpy(device_w, &w, NR_BASELINES * NR_TIMESTEPS * sizeof(float));
    q.memcpy(device_wavenumbers, &wavenumbers, NR_CHANNELS * sizeof(float));
    q.memcpy(device_spheroidal, &spheroidal, SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float));
    // q.memcpy(device_aterms, &aterms, NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float) * 8);
    q.memcpy(device_metadata, &metadata, NR_BASELINES * NR_TIMESLOTS * sizeof(int) * 9);
    // q.memcpy(device_subgrid, &subgrid, NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float) * 2);
    q.wait();
    auto end_copy_to_device = high_resolution_clock::now();

    // Run reference
    auto begin_kernel = high_resolution_clock::now();
    for (int i = 0; i <  atoi(argv[1]); i++) {
        // std::cout << "hey" << std::endl;
        kernel_gridder(
            q, NR_SUBGRIDS, GRID_SIZE, SUBGRID_SIZE, IMAGE_SIZE, W_STEP, NR_CHANNELS, NR_STATIONS,
            device_u, device_v, device_w, device_wavenumbers, visibilities, device_spheroidal,
            aterms, device_metadata, subgrid
        );
    }
    auto end_kernel = high_resolution_clock::now();

    auto begin_copy_to_host = high_resolution_clock::now();
    // q.memcpy(&subgrid, device_subgrid, NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float) * 2);
    // q.wait();
    auto end_copy_to_host = high_resolution_clock::now();

    auto create_time_host = duration_cast<nanoseconds>(end_create_host - begin_create_host).count();
    auto create_time_device = duration_cast<nanoseconds>(end_create_device - begin_create_device).count();
    auto init_time = duration_cast<nanoseconds>(end_init - begin_init).count();
    auto copy_to_device = duration_cast<nanoseconds>(end_copy_to_device - begin_copy_to_device).count();
    auto kernel_time = duration_cast<nanoseconds>(end_kernel - begin_kernel).count();
    auto copy_to_host = duration_cast<nanoseconds>(end_copy_to_host - begin_copy_to_host).count();

    std::cout << ">>> Kernel iterations: " <<  atoi(argv[1]) << std::endl;
    std::cout << ">> Object creation (host): " << create_time_host << std::endl;
    std::cout << ">> Object creation (device): " << create_time_device << std::endl;
    std::cout << ">> Object initialisation: " << init_time << std::endl;
    std::cout << ">> H2D: " << copy_to_device << std::endl;
    std::cout << ">> Kernel duration: " << kernel_time << std::endl;
    std::cout << ">> D2H: " << copy_to_host << std::endl;
    std::cout << std::endl;

    // Print subgrids for checking the gridder
    // printSubgrid(subgrid);

    // NOTE only free the device arrays
    free(device_u, q);
    free(device_v, q);
    free(device_w, q);
    free(device_wavenumbers, q);
    free(device_spheroidal, q);
    // free(device_aterms, q);
    free(device_metadata, q);
    // free(device_subgrid, q);

    return EXIT_SUCCESS;
}
