#include <complex>
#include <memory>
#include <vector>


#define GRID_SIZE            1024
#define NR_CORRELATIONS      4
#define SUBGRID_SIZE         32
#define IMAGE_SIZE           0.01f
#define W_STEP               0
#define NR_CHANNELS          16 // number of channels per subgrid
#define NR_STATIONS          10
#define NR_TIMESLOTS         4 // NOTE: increased from 2 so we can keep the gpu busy
#define NR_TIMESTEPS_SUBGRID 128 // number of timesteps per subgrid
#define NR_TIMESTEPS         (NR_TIMESTEPS_SUBGRID * NR_TIMESLOTS) // number of timesteps per baseline
#define NR_BASELINES         ((NR_STATIONS * (NR_STATIONS - 1)) / 2)
#define NR_SUBGRIDS          (NR_BASELINES * NR_TIMESLOTS)


void initialize_uvw(
    std::array<float, NR_BASELINES * NR_TIMESTEPS>& u,
    std::array<float, NR_BASELINES * NR_TIMESTEPS>& v,
    std::array<float, NR_BASELINES * NR_TIMESTEPS>& w);

void initialize_frequencies(
    std::array<double, NR_CHANNELS>& frequencies);

void initialize_wavenumbers(
    const std::array<double, NR_CHANNELS> frequencies,
    std::array<float, NR_CHANNELS>& wavenumbers);

void initialize_visibilities(
    const std::array<double, NR_CHANNELS> frequencies,
    const std::array<float, NR_BASELINES * NR_TIMESTEPS> u,
    const std::array<float, NR_BASELINES * NR_TIMESTEPS> v,
    std::vector<std::array<std::complex<float>, 4>>& visibilities);

void initialize_baselines(
    std::array<std::array<int, 2>, NR_BASELINES>& stations);

void initialize_spheroidal(
    std::array<float, SUBGRID_SIZE * SUBGRID_SIZE>& spheroidal);

void initialize_aterms(
    const std::array<float, SUBGRID_SIZE * SUBGRID_SIZE> spheroidal,
    std::array<std::array<std::complex<float>, 4>, NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE>& aterms);

void initialize_metadata(
    const std::array<std::array<int, 2>, NR_BASELINES> stations,
    std::array<std::array<int, 9>, NR_BASELINES * NR_TIMESLOTS>& metadata);

void initialize_subgrids(
    std::array<std::complex<float>, NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE>& subgrids);
