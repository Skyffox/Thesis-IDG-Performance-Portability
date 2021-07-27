#include <complex>
#include "types.h"


void kernel_gridder_buffer(
    sycl::queue q,
    const int   nr_subgrids,
    const int   grid_size,
    const int   subgrid_size,
    const float image_size,
    const float w_step_in_lambda,
    const int   nr_channels,
    const int   nr_stations,
    sycl::buffer<float, 1, sycl::detail::aligned_allocator<char>, void>                              u,
    sycl::buffer<float, 1, sycl::detail::aligned_allocator<char>, void>                              v,
    sycl::buffer<float, 1, sycl::detail::aligned_allocator<char>, void>                              w,
    sycl::buffer<float, 1, sycl::detail::aligned_allocator<char>, void>                              wavenumbers,
    sycl::buffer<std::array<std::complex<float>, 4>, 1, sycl::detail::aligned_allocator<char>, void> visibilities,
    sycl::buffer<float, 1, sycl::detail::aligned_allocator<char>, void>                              spheroidal,
    sycl::buffer<std::array<std::complex<float>, 4>, 1, sycl::detail::aligned_allocator<char>, void> aterms,
    sycl::buffer<std::array<int, 9>, 1, sycl::detail::aligned_allocator<char>, void>                 metadata,
    sycl::buffer<std::complex<float>, 1, sycl::detail::aligned_allocator<char>, void>&               subgrids);


void kernel_gridder_ref(
    const int   nr_subgrids,
    const int   gridsize,
    const int   subgridsize,
    const float imagesize,
    const float w_step_in_lambda,
    const int   nr_channels,
    const int   nr_stations,
    const idg::UVWCoordinate<float>* uvw,
    const float*                     wavenumbers,
    const std::complex<float>*       visibilities,
    const float*                     spheroidal,
    const std::complex<float>*       aterms,
    const idg::Metadata*             metadata,
          std::complex<float>*       subgrid);


void kernel_gridder_USM(
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
