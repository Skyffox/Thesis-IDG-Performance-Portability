#include <cmath>

#include "init.h"
#include "gridder.h"


void kernel_gridder_empty(
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

    const int SUBGRID_S         = NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE;
    const int UVW_SIZE          = NR_BASELINES * NR_TIMESTEPS;
    const int VISIBILITIES_SIZE = NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS * NR_CORRELATIONS;
    const int ATERMS_SIZE       = NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE * NR_CORRELATIONS;
    const int METADATA_SIZE     = NR_BASELINES * NR_TIMESLOTS * 6;
    const int SPHEROIDAL_SIZE   = SUBGRID_SIZE * SUBGRID_SIZE;

    // Iterate all subgrids
    #pragma omp target map(from: subgrid[0:SUBGRID_S]), map(to: uvw[0:UVW_SIZE], wavenumbers[0:NR_CHANNELS], visibilities[0:VISIBILITIES_SIZE], spheroidal[0:SPHEROIDAL_SIZE], aterms[0:ATERMS_SIZE], metadata[0:METADATA_SIZE])
    #pragma omp parallel for
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
    } // end s
}  // end kernel_gridder


void kernel_gridder(
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

    const int SUBGRID_S         = NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE;
    const int UVW_SIZE          = NR_BASELINES * NR_TIMESTEPS;
    const int VISIBILITIES_SIZE = NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS * NR_CORRELATIONS;
    const int ATERMS_SIZE       = NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE * NR_CORRELATIONS;
    const int METADATA_SIZE     = NR_BASELINES * NR_TIMESLOTS * 6;
    const int SPHEROIDAL_SIZE   = SUBGRID_SIZE * SUBGRID_SIZE;

    // Iterate all subgrids
    #pragma omp target map(from: subgrid[0:SUBGRID_S]), map(to: uvw[0:UVW_SIZE], wavenumbers[0:NR_CHANNELS], visibilities[0:VISIBILITIES_SIZE], spheroidal[0:SPHEROIDAL_SIZE], aterms[0:ATERMS_SIZE], metadata[0:METADATA_SIZE])
    #pragma omp parallel for
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
                apply_aterm_gridder(pixels, aterm1_ptr, aterm2_ptr);

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
