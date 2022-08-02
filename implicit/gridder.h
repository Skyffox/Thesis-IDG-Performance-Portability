#include "init.h"


void kernel_gridder_empty(
  sycl::queue q,
  const int   nr_subgrids,
  const int   grid_size,
  const int   subgrid_size,
  const float image_size,
  const float w_step_in_lambda,
  const int   nr_channels,
  const int   nr_stations,
  float                              *u,
  float                              *v,
  float                              *w,
  float                              *wavenumbers,
  std::array<std::complex<float>, 4> *visibilities,
  float                              *spheroidal,
  std::array<std::complex<float>, 4> *aterms,
  std::array<int, 9>                 *metadata,
  std::complex<float>                *subgrid);


void kernel_gridder(
  sycl::queue q,
  const int   nr_subgrids,
  const int   grid_size,
  const int   subgrid_size,
  const float image_size,
  const float w_step_in_lambda,
  const int   nr_channels,
  const int   nr_stations,
  float                              *u,
  float                              *v,
  float                              *w,
  float                              *wavenumbers,
  std::array<std::complex<float>, 4> *visibilities,
  float                              *spheroidal,
  std::array<std::complex<float>, 4> *aterms,
  std::array<int, 9>                 *metadata,
  std::complex<float>                *subgrid);
