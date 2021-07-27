#include <complex>
#include <memory>
#include <vector>

#include <cassert>

#include "types.h"


#define GRID_SIZE            1024
#define NR_CORRELATIONS      4
#define SUBGRID_SIZE         32
#define IMAGE_SIZE           0.01f
#define W_STEP               0
#define NR_CHANNELS          16 // number of channels per subgrid
#define NR_STATIONS          10
#define NR_TIMESLOTS         2
#define NR_TIMESTEPS_SUBGRID 128 // number of timesteps per subgrid
#define NR_TIMESTEPS         (NR_TIMESTEPS_SUBGRID * NR_TIMESLOTS) // number of timesteps per baseline
#define NR_BASELINES         ((NR_STATIONS * (NR_STATIONS - 1)) / 2)
#define NR_SUBGRIDS          (NR_BASELINES * NR_TIMESLOTS)


void initialize_uvw_buffer(
    std::array<float, NR_BASELINES * NR_TIMESTEPS>& u,
    std::array<float, NR_BASELINES * NR_TIMESTEPS>& v,
    std::array<float, NR_BASELINES * NR_TIMESTEPS>& w);

void initialize_frequencies_buffer(
    std::array<double, NR_CHANNELS>& frequencies);

void initialize_wavenumbers_buffer(
    const std::array<double, NR_CHANNELS> frequencies,
    std::array<float, NR_CHANNELS>& wavenumbers);

void initialize_visibilities_buffer(
    const std::array<double, NR_CHANNELS> frequencies,
    const std::array<float, NR_BASELINES * NR_TIMESTEPS> u,
    const std::array<float, NR_BASELINES * NR_TIMESTEPS> v,
    std::vector<std::array<std::complex<float>, 4>>& visibilities);

void initialize_baselines_buffer(
    std::array<std::array<int, 2>, NR_BASELINES>& stations);

void initialize_spheroidal_buffer(
    std::array<float, SUBGRID_SIZE * SUBGRID_SIZE>& spheroidal);

void initialize_aterms_buffer(
    const std::array<float, SUBGRID_SIZE * SUBGRID_SIZE> spheroidal,
    std::array<std::array<std::complex<float>, 4>, NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE>& aterms);

void initialize_metadata_buffer(
    const std::array<std::array<int, 2>, NR_BASELINES> stations,
    std::array<std::array<int, 9>, NR_BASELINES * NR_TIMESLOTS>& metadata);

void initialize_subgrids_buffer(
    std::vector<std::complex<float>>& subgrid);

// --------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------

void initialize_uvw_ref(
    unsigned int grid_size,
    idg::Array2D<idg::UVWCoordinate<float>>& uvw);

void initialize_frequencies_ref(
    idg::Array1D<float>& frequencies);

void initialize_wavenumbers_ref(
    const idg::Array1D<float>& frequencies,
    idg::Array1D<float>& wavenumbers);

void initialize_visibilities_ref(
    unsigned int grid_size,
    float image_size,
    const idg::Array1D<float>& frequencies,
    const idg::Array2D<idg::UVWCoordinate<float>>& uvw,
    idg::Array3D<idg::Visibility<std::complex<float>>>& visibilities);

void initialize_baselines_ref(
    unsigned int nr_stations,
    idg::Array1D<idg::Baseline>& baselines);

void initialize_spheroidal_ref(
    idg::Array2D<float>& spheroidal);

void initialize_aterms_ref(
    const idg::Array2D<float>& spheroidal,
    idg::Array4D<idg::Matrix2x2<std::complex<float>>>& aterms);

void initialize_metadata_ref(
    unsigned int grid_size,
    unsigned int nr_timeslots,
    unsigned int nr_timesteps_subgrid,
    const idg::Array1D<idg::Baseline>& baselines,
    idg::Array1D<idg::Metadata>& metadata);

void initialize_subgrids_ref(
	idg::Array4D<std::complex<float>>& subgrids);

// --------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------

void initialize_uvw_implicit(
    float *u,
    float *v,
    float *w);

void initialize_frequencies_implicit(
    double *frequencies);

void initialize_wavenumbers_implicit(
    double *frequencies,
    float *wavenumbers);

void initialize_visibilities_implicit(
    double *frequencies,
    float *u,
    float *v,
    std::array<std::complex<float>, 4> *visibilities);

void initialize_baselines_implicit(
    std::array<int, 2> *stations);

void initialize_spheroidal_implicit(
    float *spheroidal);

void initialize_aterms_implicit(
    float *spheroidal,
    std::array<std::complex<float>, 4> *aterms);

void initialize_metadata_implicit(
    std::array<int, 2> *stations,
    std::array<int, 9> *metadata);

void initialize_subgrids_implicit(
    std::complex<float> *subgrid);
