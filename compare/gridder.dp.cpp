#include <CL/sycl.hpp>

#include "gridder.h"
#include "math.h"

#include <iostream>
#include <cmath>

using namespace sycl;


void kernel_gridder_buffer(
    queue       q,
    const int   nr_subgrids,
    const int   grid_size,
    const int   subgrid_size,
    const float image_size,
    const float w_step_in_lambda,
    const int   nr_channels,
    const int   nr_stations,
    sycl::buffer<float, 1, sycl::detail::aligned_allocator<char>, void> UCoordinate,
    sycl::buffer<float, 1, sycl::detail::aligned_allocator<char>, void> VCoordinate,
    sycl::buffer<float, 1, sycl::detail::aligned_allocator<char>, void> WCoordinate,
    sycl::buffer<float, 1, sycl::detail::aligned_allocator<char>, void> wavenumbers,
    sycl::buffer<std::array<std::complex<float>, 4>, 1, sycl::detail::aligned_allocator<char>, void> visibilities,
    sycl::buffer<float, 1, sycl::detail::aligned_allocator<char>, void> spheroidal,
    sycl::buffer<std::array<std::complex<float>, 4>, 1, sycl::detail::aligned_allocator<char>, void> aterms,
    sycl::buffer<std::array<int, 9>, 1, sycl::detail::aligned_allocator<char>, void> metadata,
    sycl::buffer<std::complex<float>, 1, sycl::detail::aligned_allocator<char>, void>& subgrids)
{
    // Iterate all subgrids
    q.submit([&](handler &h) {
        // Make stuff faster if the compiler knows how we want to access the data
        auto UCoordinate_acc = UCoordinate.get_access<access::mode::read>(h);
        auto VCoordinate_acc = VCoordinate.get_access<access::mode::read>(h);
        auto WCoordinate_acc = WCoordinate.get_access<access::mode::read>(h);
        auto wavenumbers_acc = wavenumbers.get_access<access::mode::read>(h);
        auto visibilities_acc = visibilities.get_access<access::mode::read>(h);
        auto spheroidal_acc = spheroidal.get_access<access::mode::read>(h);
        auto aterms_acc = aterms.get_access<access::mode::read>(h);
        auto metadata_acc = metadata.get_access<access::mode::read>(h);
        auto subgrid_acc = subgrids.get_access<access::mode::write>(h);

        h.parallel_for(nr_subgrids, [=](id<1> s) {
            const std::array<int, 9> m = metadata_acc[s];
            const int time_offset      = (m[0] - metadata_acc[0][0]) + m[1];
            const int nr_timesteps     = m[2];
            const int aterm_index      = m[3];
            const int station1         = m[4];
            const int station2         = m[5];
            const int x_coordinate     = m[6];
            const int y_coordinate     = m[7];
            const float w_offset_in_lambda = w_step_in_lambda * (m[8] + 0.5);

            // Compute u and v offset in wavelengths
            const float u_offset = (x_coordinate + subgrid_size / 2 - grid_size / 2) * (2 * M_PI / image_size);
            const float v_offset = (y_coordinate + subgrid_size / 2 - grid_size / 2) * (2 * M_PI / image_size);
            const float w_offset = 2 * M_PI * w_offset_in_lambda;

            // Iterate all pixels in subgrid
            for (int y = 0; y < subgrid_size; y++) {
                for (int x = 0; x < subgrid_size; x++) {
                    std::array<std::complex<float>, NR_POLARIZATIONS> pixels;

                    float l = compute_l(x, subgrid_size, image_size);
                    float m = compute_m(y, subgrid_size, image_size);
                    float n = compute_n(l, m);

                    for (int time = 0; time < nr_timesteps; time++) {
                        // Load UVW coordinates
                        float uPoint = UCoordinate_acc[time_offset + time];
                        float vPoint = VCoordinate_acc[time_offset + time];
                        float wPoint = WCoordinate_acc[time_offset + time];

                        // Compute phase index
                        float phase_index = uPoint * l + vPoint * m + wPoint * n;

                        // Compute phase offset
                        float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                        // Update pixel for every channel
                        for (int chan = 0; chan < nr_channels; chan++) {
                            // Compute phase
                            float phase = phase_offset - (phase_index * wavenumbers_acc[chan]);

                            // Compute phasor
                            std::complex<float> phasor = {cosf(phase), sinf(phase)};

                            // Update pixel for every polarization
                            size_t index = (time_offset + time) * nr_channels + chan;
                            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                                std::complex<float> visibility = visibilities_acc[index][pol];
                                pixels[pol] += visibility * phasor;
                            }
                        } // end for chan
                    } // end for time

                    // Load a term for station1
                    int station1_index = (aterm_index * nr_stations + station1) *
                                          subgrid_size * subgrid_size +
                                          y * subgrid_size + x;
                    const std::array<std::complex<float>, 4> aterm1 = aterms_acc[station1_index];

                    // Load a term for station2
                    int station2_index = (aterm_index * nr_stations + station2) *
                                          subgrid_size * subgrid_size +
                                          y * subgrid_size + x;
                    const std::array<std::complex<float>, 4> aterm2 = aterms_acc[station2_index];

                    // Apply aterm
                    apply_aterm_gridder(pixels, aterm1, aterm2);

                    // Load spheroidal
                    float sph = spheroidal_acc[y * subgrid_size + x];

                    // Set subgrid value
                    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        unsigned idx_subgrid = s * NR_POLARIZATIONS * subgrid_size * subgrid_size +
                                               pol * subgrid_size * subgrid_size +
                                               y * subgrid_size + x;
                        subgrid_acc[idx_subgrid] = pixels[pol] * sph;
                    }
                } // x
            } // y
        }); // end parallel_for
    }); // end submit
    q.wait(); // Have to invoke a wait to copy back, believe this is because of the vector in use.
}  // end kernel_gridder


//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void kernel_gridder_ref(
    const int   nr_subgrids,
    const int   grid_size,
    const int   subgrid_size,
    const float image_size,
    const float w_step_in_lambda,
    const int   nr_channels,
    const int   nr_stations,
    const idg::UVWCoordinate<float>* uvw,
    const float*                     wavenumbers,
    const std::complex<float>*       visibilities,
    const float*                     spheroidal,
    const std::complex<float>*       aterms,
    const idg::Metadata*             metadata,
          std::complex<float>*       subgrid)
{
    // Find offset of first subgrid
    const idg::Metadata m       = metadata[0];
    const int baseline_offset_1 = m.baseline_offset;

    // Iterate all subgrids
    for (int s = 0; s < nr_subgrids; s++) {
        // Load metadata
        const idg::Metadata m  = metadata[s];
        const int time_offset  = (m.baseline_offset - baseline_offset_1) + m.time_offset;
        const int nr_timesteps = m.nr_timesteps;
        const int aterm_index  = m.aterm_index;
        const int station1     = m.baseline.station1;
        const int station2     = m.baseline.station2;
        const int x_coordinate = m.coordinate.x;
        const int y_coordinate = m.coordinate.y;
        const float w_offset_in_lambda = w_step_in_lambda * (m.coordinate.z + 0.5);

        // Compute u and v offset in wavelenghts
        const float u_offset = (x_coordinate + subgrid_size/2 - grid_size/2) * (2*M_PI / image_size);
        const float v_offset = (y_coordinate + subgrid_size/2 - grid_size/2) * (2*M_PI / image_size);
        const float w_offset = 2*M_PI * w_offset_in_lambda;

        // Iterate all pixels in subgrid
        for (int y = 0; y < subgrid_size; y++) {
            for (int x = 0; x < subgrid_size; x++) {
                // Initialize pixel for every polarization
                std::complex<float> pixels[NR_POLARIZATIONS];
                memset(pixels, 0, NR_POLARIZATIONS * sizeof(std::complex<float>));

                // Compute l,m,n
                float l = compute_l(x, subgrid_size, image_size);
                float m = compute_m(y, subgrid_size, image_size);
                float n = compute_n(l, m);

                // Iterate all timesteps
                for (int time = 0; time < nr_timesteps; time++) {
                    // Load UVW coordinates
                    float u = uvw[time_offset + time].u;
                    float v = uvw[time_offset + time].v;
                    float w = uvw[time_offset + time].w;

                    // Compute phase index
                    float phase_index = u*l + v*m + w*n;

                    // Compute phase offset
                    float phase_offset = u_offset*l + v_offset*m + w_offset*n;

                    // Update pixel for every channel
                    for (int chan = 0; chan < nr_channels; chan++) {
                        // Compute phase
                        float phase = phase_offset - (phase_index * wavenumbers[chan]);

                        // Compute phasor
                        std::complex<float> phasor = {cosf(phase), sinf(phase)};

                        // Update pixel for every polarization
                        size_t index = (time_offset + time)*nr_channels + chan;
                        for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                            std::complex<float> visibility = visibilities[index * NR_POLARIZATIONS + pol];
                            pixels[pol] += visibility * phasor;
                        }
                    } // end for chan
                } // end for time

                // Load a term for station1
                int station1_index =
                    (aterm_index * nr_stations + station1) *
                    subgrid_size * subgrid_size * NR_POLARIZATIONS +
                    y * subgrid_size * NR_POLARIZATIONS + x * NR_POLARIZATIONS;
                const std::complex<float>* aterm1_ptr = &aterms[station1_index];

                // Load aterm for station2
                int station2_index =
                    (aterm_index * nr_stations + station2) *
                    subgrid_size * subgrid_size * NR_POLARIZATIONS +
                    y * subgrid_size * NR_POLARIZATIONS + x * NR_POLARIZATIONS;
                const std::complex<float>* aterm2_ptr = &aterms[station2_index];

                // Apply aterm
                apply_aterm_gridder_ref(pixels, aterm1_ptr, aterm2_ptr);

                // Load spheroidal
                float sph = spheroidal[y * subgrid_size + x];

                // Set subgrid value
                for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                    unsigned idx_subgrid =
                        s * NR_POLARIZATIONS * subgrid_size * subgrid_size +
                        pol * subgrid_size * subgrid_size + y * subgrid_size + x;
                    subgrid[idx_subgrid] = pixels[pol] * sph;
                }
            } // end x
        } // end y
    } // end s
}  // end kernel_gridder

//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------

void kernel_gridder_USM(
    queue       q,
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
    std::complex<float>                *subgrid)

{
    // Find offset of first subgrid
    const int baseline_offset_1 = metadata[0][0];

    // Iterate all subgrids
    q.submit([&](handler &h) {
        h.parallel_for(nr_subgrids, [=](id<1> s) {
            const std::array<int, 9> m = metadata[s];
            const int time_offset      = (m[0] - baseline_offset_1) + m[1];
            const int nr_timesteps     = m[2];
            const int aterm_index      = m[3];
            const int station1         = m[4];
            const int station2         = m[5];
            const int x_coordinate     = m[6];
            const int y_coordinate     = m[7];
            const float w_offset_in_lambda = w_step_in_lambda * (m[8] + 0.5);

            // Compute u and v offset in wavelengths
            const float u_offset = (x_coordinate + subgrid_size / 2 - grid_size / 2) * (2 * M_PI / image_size);
            const float v_offset = (y_coordinate + subgrid_size / 2 - grid_size / 2) * (2 * M_PI / image_size);
            const float w_offset = 2 * M_PI * w_offset_in_lambda;

            // Iterate all pixels in subgrid
            for (int y = 0; y < subgrid_size; y++) {
                for (int x = 0; x < subgrid_size; x++) {
                    std::array<std::complex<float>, NR_POLARIZATIONS> pixels;

                    float l = compute_l(x, subgrid_size, image_size);
                    float m = compute_m(y, subgrid_size, image_size);
                    float n = compute_n(l, m);

                    for (int time = 0; time < nr_timesteps; time++) {
                        // Load UVW coordinates
                        float uPoint = u[time_offset + time];
                        float vPoint = v[time_offset + time];
                        float wPoint = w[time_offset + time];

                        // Compute phase index
                        float phase_index = uPoint * l + vPoint * m + wPoint * n;

                        // Compute phase offset
                        float phase_offset = u_offset * l + v_offset * m + w_offset * n;

                        // Update pixel for every channel
                        for (int chan = 0; chan < nr_channels; chan++) {
                            // Compute phase
                            float phase = phase_offset - (phase_index * wavenumbers[chan]);

                            // Compute phasor
                            std::complex<float> phasor = {cosf(phase), sinf(phase)};

                            // Update pixel for every polarization
                            size_t index = (time_offset + time) * nr_channels + chan;
                            for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                                std::complex<float> visibility = visibilities[index][pol];
                                pixels[pol] += visibility * phasor;
                            }
                        } // end for chan
                    } // end for time

                    // Load a term for station1
                    int station1_index = (aterm_index * nr_stations + station1) *
                                          subgrid_size * subgrid_size +
                                          y * subgrid_size + x;
                    const std::array<std::complex<float>, 4> aterm1 = aterms[station1_index];

                    // Load a term for station2
                    int station2_index = (aterm_index * nr_stations + station2) *
                                          subgrid_size * subgrid_size +
                                          y * subgrid_size + x;
                    const std::array<std::complex<float>, 4> aterm2 = aterms[station2_index];

                    // Apply aterm
                    apply_aterm_gridder(pixels, aterm1, aterm2);

                    // Load spheroidal
                    float sph = spheroidal[y * subgrid_size + x];

                    // Set subgrid value
                    for (int pol = 0; pol < NR_POLARIZATIONS; pol++) {
                        unsigned idx_subgrid = s * NR_POLARIZATIONS * subgrid_size * subgrid_size +
                                               pol * subgrid_size * subgrid_size +
                                               y * subgrid_size + x;
                        subgrid[idx_subgrid] = pixels[pol] * sph;
                    }
                } // x
            } // y
        }); // end parallel_for
    }); // end submit
    q.wait();
}  // end kernel_gridder
