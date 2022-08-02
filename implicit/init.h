#include <complex>
#include <memory>
#include <vector>


#define GRID_SIZE            1024
#define NR_CORRELATIONS      4
#define SUBGRID_SIZE         32
#define IMAGE_SIZE           0.01f
#define W_STEP               0
#define NR_CHANNELS          16 // number of channels per subgrid
#define NR_STATIONS          48 // NOTE: small param uses 10 and big param 48
#define NR_TIMESLOTS         4 // NOTE: small param uses 2 and big param 4
#define NR_TIMESTEPS_SUBGRID 128 // number of timesteps per subgrid
#define NR_TIMESTEPS         (NR_TIMESTEPS_SUBGRID * NR_TIMESLOTS) // number of timesteps per baseline
#define NR_BASELINES         ((NR_STATIONS * (NR_STATIONS - 1)) / 2)
#define NR_SUBGRIDS          (NR_BASELINES * NR_TIMESLOTS)


void initialize_uvw(
    float *u,
    float *v,
    float *w);

void initialize_frequencies(
    double *frequencies);

void initialize_wavenumbers(
    const double *frequencies,
    float *wavenumbers);

void initialize_visibilities(
    const double *frequencies,
    const float *u,
    const float *v,
    std::array<std::complex<float>, 4> *visibilities);

void initialize_baselines(
    std::array<int, 2> *stations);

void initialize_spheroidal(
    float *spheroidal);

void initialize_aterms(
    const float *spheroidal,
    std::array<std::complex<float>, 4> *aterms);

void initialize_metadata(
    const std::array<int, 2> *stations,
    std::array<int, 9> *metadata);

void initialize_subgrids(
    std::complex<float> *subgrid);
