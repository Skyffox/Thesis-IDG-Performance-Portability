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
    std::cout << ">>> Selected device: " << dev.get_info<info::device::name>() << "\n";
}


int main(int argc, char **argv)
{
    // if (argc != 2) {
    //     printf("Usage: ./program kernel_iterations\n");
    //     exit(0);
    // }

    queue q( default_selector{} );
    output_dev_info(device{ default_selector{} });

    // NOTE: Because we test for a large amount of timeslots/stations at one point we decided to convert all big arrays
    // that are affected to vectors as we would otherwise have a Stack Overflow. As a results the following measurement
    // Takes a lot longer because std::vector is immediately initialized.
    auto begin_create = steady_clock::now();
    std::vector<float> u(NR_BASELINES * NR_TIMESTEPS);
    std::vector<float> v(NR_BASELINES * NR_TIMESTEPS);
    std::vector<float> w(NR_BASELINES * NR_TIMESTEPS);
    std::array<double, NR_CHANNELS> frequencies;
    std::array<float, NR_CHANNELS> wavenumbers;
    std::vector<std::array<std::complex<float>, 4>> visibilities(NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS);
    std::array<std::array<int, 2>, NR_BASELINES> stations;
    std::array<float, SUBGRID_SIZE * SUBGRID_SIZE> spheroidal;
    std::vector<std::array<std::complex<float>, 4>> aterms(NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE);
    std::array<std::array<int, 9>, NR_BASELINES * NR_TIMESLOTS> metadata;
    std::vector<std::complex<float>> subgrids(NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE);
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
    initialize_subgrids(subgrids);
    auto end_init = steady_clock::now();

    auto begin_buffer = steady_clock::now();
    buffer UCoordinate_buf(u);
    buffer VCoordinate_buf(v);
    buffer WCoordinate_buf(w);
    buffer wavenumbers_buf(wavenumbers);
    buffer visibilities_buf(visibilities);
    buffer spheroidal_buf(spheroidal);
    buffer aterms_buf(aterms);
    buffer metadata_buf(metadata);
    buffer subgrids_buf(subgrids);
    auto end_buffer = steady_clock::now();

    // printUVW(u, v, w);               // NOTE: Prints a lot
    // printFrequencies(frequencies);
    // printWavenumbers(wavenumbers);
    // printVisibilities(visibilities); // NOTE: Prints a lot
    // printBaselines(stations);
    // printSpheroidal(spheroidal);
    // printAterms(aterms);             // NOTE: Prints a lot
    // printMetadata(metadata);
    // printSubgrid(subgrids);          // NOTE: Prints a lot

    // WARMUP
    kernel_gridder(
        q, NR_SUBGRIDS, GRID_SIZE, SUBGRID_SIZE, IMAGE_SIZE, W_STEP, NR_CHANNELS, NR_STATIONS,
        UCoordinate_buf, VCoordinate_buf, WCoordinate_buf, wavenumbers_buf,
        visibilities_buf, spheroidal_buf, aterms_buf, metadata_buf, subgrids_buf
    );

    // Run the buffer gridder
    auto begin_kernel = steady_clock::now();
    // for (int i = 0; i < atoi(argv[1]); i++) {
    kernel_gridder(
        q, NR_SUBGRIDS, GRID_SIZE, SUBGRID_SIZE, IMAGE_SIZE, W_STEP, NR_CHANNELS, NR_STATIONS,
        UCoordinate_buf, VCoordinate_buf, WCoordinate_buf, wavenumbers_buf,
        visibilities_buf, spheroidal_buf, aterms_buf, metadata_buf, subgrids_buf
    );
    // }
    subgrids_buf.get_access<access::mode::read>(); // <--- Host Accessor to Synchronize Memory
    auto end_kernel = steady_clock::now();

    auto create_time = duration_cast<nanoseconds>(end_create - begin_create).count();
    auto init_time = duration_cast<nanoseconds>(end_init - begin_init).count();
    auto buffer_time = duration_cast<nanoseconds>(end_buffer - begin_buffer).count();
    auto kernel_time = duration_cast<nanoseconds>(end_kernel - begin_kernel).count();

    // std::cout << ">> Kernel iterations: " << atoi(argv[1]) << std::endl;
    std::cout << ">> Object creation: " << create_time << std::endl;
    std::cout << ">> Object initialisation: " << init_time << std::endl;
    std::cout << ">> Buffer: " << buffer_time << std::endl;
    std::cout << ">> Kernel duration: " << kernel_time << std::endl;
    std::cout << std::endl;

    return EXIT_SUCCESS;
}
