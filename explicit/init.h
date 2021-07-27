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
#define NR_TIMESLOTS         2
#define NR_TIMESTEPS_SUBGRID 128 // number of timesteps per subgrid
#define NR_TIMESTEPS         (NR_TIMESTEPS_SUBGRID * NR_TIMESLOTS) // number of timesteps per baseline
#define NR_BASELINES         ((NR_STATIONS * (NR_STATIONS - 1)) / 2)
#define NR_SUBGRIDS          (NR_BASELINES * NR_TIMESLOTS)


void initialize_uvw(
    float u[NR_BASELINES * NR_TIMESTEPS],
    float v[NR_BASELINES * NR_TIMESTEPS],
    float w[NR_BASELINES * NR_TIMESTEPS]);

void initialize_frequencies(
    double frequencies[NR_CHANNELS]);

void initialize_wavenumbers(
    double frequencies[NR_CHANNELS],
    float wavenumbers[NR_CHANNELS]);

void initialize_visibilities(
    double frequencies[NR_CHANNELS],
    float u[NR_BASELINES * NR_TIMESTEPS],
    float v[NR_BASELINES * NR_TIMESTEPS],
    std::array<std::complex<float>, 4> *visibilities);

void initialize_baselines(
    std::array<int, 2> stations[NR_BASELINES]);

void initialize_spheroidal(
    float spheroidal[SUBGRID_SIZE * SUBGRID_SIZE]);

void initialize_aterms(
    float spheroidal[SUBGRID_SIZE * SUBGRID_SIZE],
    std::array<std::complex<float>, 4> *aterms);

void initialize_metadata(
    std::array<int, 2> stations[NR_BASELINES],
    std::array<int, 9> metadata[NR_BASELINES * NR_TIMESLOTS]);

void initialize_subgrids(
    std::complex<float> *subgrid);
