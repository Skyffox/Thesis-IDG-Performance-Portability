// OPNEAPI LIBS
#include <CL/sycl.hpp>

// EXTRA LIBS
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>

// THEIR LIBS
#include "init.h"
#include "gridder.h"
#include "print.h"

// Some calls also exist in the std library so we only use one namespace to avoid ambiguity
using namespace sycl;


// NOTE: can only test one memory model at the time
int main(int argc, char **argv) {
    // Create queue to push work to an accelerator
    queue q( cpu_selector{} );

    // NOTE: buffer
    std::array<float, NR_BASELINES * NR_TIMESTEPS> u;
    std::array<float, NR_BASELINES * NR_TIMESTEPS> v;
    std::array<float, NR_BASELINES * NR_TIMESTEPS> w;
    std::array<double, NR_CHANNELS> frequencies;
    std::array<float, NR_CHANNELS> wavenumbers;
    std::vector<std::array<std::complex<float>, 4>> visibilities(NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS);
    std::array<std::array<int, 2>, NR_BASELINES> stations;
    std::array<float, SUBGRID_SIZE * SUBGRID_SIZE> spheroidal;
    std::array<std::array<std::complex<float>, 4>, NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE> aterms;
    std::array<std::array<int, 9>, NR_BASELINES * NR_TIMESLOTS> metadata;

    // NOTE: we allocated too much on the stack so this is now allocated on the heap
    std::vector<std::complex<float>> subgrids(NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE);

    // NOTE: ref
    idg::Array2D<idg::UVWCoordinate<float>> uvw_ref(NR_BASELINES, NR_TIMESTEPS);
    idg::Array1D<float> frequencies_ref(NR_CHANNELS);
    idg::Array1D<float> wavenumbers_ref(NR_CHANNELS);
    idg::Array3D<idg::Visibility<std::complex<float>>> visibilities_ref(NR_BASELINES, NR_TIMESTEPS, NR_CHANNELS);
    idg::Array1D<idg::Baseline> baselines_ref(NR_BASELINES);
    idg::Array2D<float> spheroidal_ref(SUBGRID_SIZE, SUBGRID_SIZE);
    idg::Array4D<idg::Matrix2x2<std::complex<float>>> aterms_ref(NR_TIMESLOTS, NR_STATIONS, SUBGRID_SIZE, SUBGRID_SIZE);
    idg::Array1D<idg::Metadata> metadata_ref(NR_SUBGRIDS);
    idg::Array4D<std::complex<float>> subgrids_ref(NR_SUBGRIDS, NR_CORRELATIONS, SUBGRID_SIZE, SUBGRID_SIZE);

    // NOTE: implicit - shared
    float *u_impl = (float *) malloc_shared(NR_BASELINES * NR_TIMESTEPS * sizeof(float), q);
    float *v_impl = (float *) malloc_shared(NR_BASELINES * NR_TIMESTEPS * sizeof(float), q);
    float *w_impl = (float *) malloc_shared(NR_BASELINES * NR_TIMESTEPS * sizeof(float), q);
    double *frequencies_impl = (double *) malloc_shared(NR_CHANNELS * sizeof(double), q);
    float *wavenumbers_impl = (float *) malloc_shared(NR_CHANNELS * sizeof(float), q);
    std::array<std::complex<float>, 4> *visibilities_impl = (std::array<std::complex<float>, 4> *) malloc_shared(NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS * sizeof(float) * 8, q);
    std::array<int, 2> *stations_impl = (std::array<int, 2> *) malloc_shared(NR_BASELINES * sizeof(int) * 2, q);
    float *spheroidal_impl = (float *) malloc_shared(SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float), q);
    std::array<std::complex<float>, 4> *aterms_impl = (std::array<std::complex<float>, 4> *) malloc_shared(NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float) * 8, q);
    std::array<int, 9> *metadata_impl = (std::array<int, 9> *) malloc_shared(NR_BASELINES * NR_TIMESLOTS * sizeof(int) * 9, q);
    std::complex<float> *subgrid_impl = (std::complex<float> *) malloc_shared(NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE * sizeof(float) * 2, q);

    // -----------------------------------------------------------------------------------------------------------------

    srand(0);
    initialize_uvw_buffer(u, v, w);
    initialize_frequencies_buffer(frequencies);
    initialize_wavenumbers_buffer(frequencies, wavenumbers);
    initialize_visibilities_buffer(frequencies, u, v, visibilities);
    initialize_baselines_buffer(stations);
    initialize_spheroidal_buffer(spheroidal);
    initialize_aterms_buffer(spheroidal, aterms);
    initialize_metadata_buffer(stations, metadata);
    initialize_subgrids_buffer(subgrids);

    srand(0);
    initialize_uvw_ref(GRID_SIZE, uvw_ref);
    initialize_frequencies_ref(frequencies_ref);
    initialize_wavenumbers_ref(frequencies_ref, wavenumbers_ref);
    initialize_visibilities_ref(GRID_SIZE, IMAGE_SIZE, frequencies_ref, uvw_ref, visibilities_ref);
    initialize_baselines_ref(NR_STATIONS, baselines_ref);
    initialize_spheroidal_ref(spheroidal_ref);
    initialize_aterms_ref(spheroidal_ref, aterms_ref);
    initialize_metadata_ref(GRID_SIZE, NR_TIMESLOTS, NR_TIMESTEPS_SUBGRID, baselines_ref, metadata_ref);
    initialize_subgrids_ref(subgrids_ref);

    srand(0);
    initialize_uvw_implicit(u_impl, v_impl, w_impl);
    initialize_frequencies_implicit(frequencies_impl);
    initialize_wavenumbers_implicit(frequencies_impl, wavenumbers_impl);
    initialize_visibilities_implicit(frequencies_impl, u_impl, v_impl, visibilities_impl);
    initialize_baselines_implicit(stations_impl);
    initialize_spheroidal_implicit(spheroidal_impl);
    initialize_aterms_implicit(spheroidal_impl, aterms_impl);
    initialize_metadata_implicit(stations_impl, metadata_impl);
    initialize_subgrids_implicit(subgrid_impl);

    // -----------------------------------------------------------------------------------------------------------------

    // // NOTE: check whether buffer input is the same
    // compare_UVW_buffer(u, v, w, uvw_ref);
    // compare_frequencies_buffer(frequencies, frequencies_ref);
    // compare_wavenumbers_buffer(wavenumbers, wavenumbers_ref); // NOTE: 6 differences
    // compare_visibilities_buffer(visibilities, visibilities_ref); // NOTE: nothing equal
    // compare_baselines_buffer(stations, baselines_ref);
    // compare_spheroidal_buffer(spheroidal, spheroidal_ref);
    // compare_aterms_buffer(aterms, aterms_ref);
    // compare_metadata_buffer(metadata, metadata_ref);
    // compare_subgrid_buffer(subgrids, subgrids_ref);
    //
    // // NOTE: check whether implicit input is the same
    // compare_UVW_implicit(u_impl, v_impl, w_impl, uvw_ref);
    // compare_frequencies_implicit(frequencies_impl, frequencies_ref);
    // compare_wavenumbers_implicit(wavenumbers_impl, wavenumbers_ref); // NOTE: 6 differences
    // compare_visibilities_implicit(visibilities_impl, visibilities_ref); // NOTE: nothing equal
    // compare_baselines_implicit(stations_impl, baselines_ref);
    // compare_spheroidal_implicit(spheroidal_impl, spheroidal_ref);
    // compare_aterms_implicit(aterms_impl, aterms_ref);
    // compare_metadata_implicit(metadata_impl, metadata_ref);
    // compare_subgrid_implicit(subgrid_impl, subgrids_ref);

    // -----------------------------------------------------------------------------------------------------------------

    // Moving data to the buffers
    buffer UCoordinate_buf(u);
    buffer VCoordinate_buf(v);
    buffer WCoordinate_buf(w);
    buffer wavenumbers_buf(wavenumbers);
    buffer visibilities_buf(visibilities);
    buffer spheroidal_buf(spheroidal);
    buffer aterms_buf(aterms);
    buffer metadata_buf(metadata);
    buffer subgrids_buf(subgrids);

    // -----------------------------------------------------------------------------------------------------------------

    kernel_gridder_buffer(
        q, NR_SUBGRIDS, GRID_SIZE, SUBGRID_SIZE, IMAGE_SIZE, W_STEP, NR_CHANNELS, NR_STATIONS,
        UCoordinate_buf, VCoordinate_buf, WCoordinate_buf, wavenumbers_buf,
        visibilities_buf, spheroidal_buf, aterms_buf, metadata_buf, subgrids_buf
    );

    kernel_gridder_ref(
        NR_SUBGRIDS, GRID_SIZE, SUBGRID_SIZE, IMAGE_SIZE, W_STEP, NR_CHANNELS, NR_STATIONS,
        uvw_ref.data(), wavenumbers_ref.data(), (std::complex<float> *) visibilities_ref.data(),
        (float *) spheroidal_ref.data(), (std::complex<float> *) aterms_ref.data(), metadata_ref.data(),
        subgrids_ref.data()
    );

    kernel_gridder_USM(
        q, NR_SUBGRIDS, GRID_SIZE, SUBGRID_SIZE, IMAGE_SIZE, W_STEP, NR_CHANNELS, NR_STATIONS,
        u_impl, v_impl, w_impl, wavenumbers_impl, visibilities_impl, spheroidal_impl, aterms_impl,
        metadata_impl, subgrid_impl
    );

    // -----------------------------------------------------------------------------------------------------------------

    std::cout << "Compare buffer with reference subgrids:" << std::endl;
    get_accuracy_buffer(subgrids, subgrids_ref);

    std::cout << std::endl;
    
    std::cout << "Compare implicit with reference subgrids:" << std::endl;
    get_accuracy_implicit(subgrid_impl, subgrids_ref);

    // -----------------------------------------------------------------------------------------------------------------

    return EXIT_SUCCESS;
}
