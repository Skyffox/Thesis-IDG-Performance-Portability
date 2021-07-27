#include <cassert>
#include <iostream>

#include "init.h"
#include "types.h"


void compare_UVW_buffer(
    std::array<float, NR_BASELINES * NR_TIMESTEPS> u_arr,
    std::array<float, NR_BASELINES * NR_TIMESTEPS> v_arr,
    std::array<float, NR_BASELINES * NR_TIMESTEPS> w_arr,
    idg::Array2D<idg::UVWCoordinate<float>>& uvw_ref)
{
    for (unsigned i = 0; i < NR_BASELINES; i++) {
        for (unsigned j = 0; j < NR_TIMESTEPS; j++) {
            float u = u_arr[i * NR_TIMESTEPS + j];
            float v = v_arr[i * NR_TIMESTEPS + j];
            float w = w_arr[i * NR_TIMESTEPS + j];

            float u_ref = uvw_ref(i, j).u;
            float v_ref = uvw_ref(i, j).v;
            float w_ref = uvw_ref(i, j).w;

            if ((abs(u - u_ref) != 0) || (abs(v - v_ref) != 0) || (abs(w - w_ref) != 0)) {
                std::cout << "This is uvw Baselines, Timestep: " << i << " " << j << std::endl;
            }
        }
    }
}

void compare_frequencies_buffer(
    std::array<double, NR_CHANNELS> frequencies,
    idg::Array1D<float>& frequencies_ref)
{
    for (unsigned i = 0; i < NR_CHANNELS; i++) {
        float freq = frequencies[i];
        float freq_ref = frequencies_ref(i);

        if (abs(freq - freq_ref) != 0) {
            std::cout << "This is frequencies Channel: " << i << std::endl;
        }
    }
}

void compare_wavenumbers_buffer(
    std::array<float, NR_CHANNELS> wavenumbers,
    idg::Array1D<float>& wavenumbers_ref)
{
    for (unsigned i = 0; i < NR_CHANNELS; i++) {
        float wav = wavenumbers[i];
        float wav_ref = wavenumbers_ref(i);

        if (abs(wav - wav_ref) != 0) {
            std::cout.precision(17);
            std::cout << "This is wavenumbers Channel: " << i << std::endl;
            std::cout << std::fixed << wav << " " << wav_ref << std::endl;
            std::cout << std::fixed << abs(wav - wav_ref) << std::endl;
            std::cout << std::endl;
        }
    }
}

void compare_visibilities_buffer(
    std::vector<std::array<std::complex<float>, 4>> visibilities,
    idg::Array3D<idg::Visibility<std::complex<float>>>& visibilities_ref)
{
    int counter = 0;
    for (unsigned bl = 0; bl < NR_BASELINES; bl++) {
        for (unsigned time = 0; time < NR_TIMESTEPS; time++) {
            for (unsigned chan = 0; chan < NR_CHANNELS; chan++) {
                std::array<std::complex<float>, 4> vis       = visibilities[bl * NR_TIMESTEPS * NR_CHANNELS + time * NR_CHANNELS + chan];
                idg::Visibility<std::complex<float>> vis_ref = visibilities_ref(bl, time, chan);

                // NOTE: not exactly same which is strange...
                float tol = 0;

                if ((abs(vis[0] - vis_ref.xx) > tol) ||
                    (abs(vis[1] - vis_ref.xy) > tol) ||
                    (abs(vis[2] - vis_ref.yx) > tol) ||
                    (abs(vis[3] - vis_ref.yy) > tol)) {
                    std::cout << "This is visibilities Baseline, Timestep, Channel: " << bl << " " << time << " " << chan << std::endl;
                    std::cout.precision(17);
                    std::cout << std::fixed << abs(vis[0].real() - vis_ref.xx.real()) << " " << abs(vis[0].imag() - vis_ref.xx.imag()) << std::endl;
                    std::cout << std::fixed << abs(vis[1].real() - vis_ref.xy.real()) << " " << abs(vis[1].imag() - vis_ref.xy.imag()) << std::endl;
                    std::cout << std::fixed << abs(vis[2].real() - vis_ref.yx.real()) << " " << abs(vis[2].imag() - vis_ref.yx.imag()) << std::endl;
                    std::cout << std::fixed << abs(vis[3].real() - vis_ref.yy.real()) << " " << abs(vis[3].imag() - vis_ref.yy.imag()) << std::endl;

                    counter += 1;
                }
            }
        }
    }
    // std::cout << "Total not right: " << counter << std::endl;
    // std::cout << "Total length: " << (NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS) << std::endl;
}

void compare_baselines_buffer(
    std::array<std::array<int, 2>, NR_BASELINES> stations,
    idg::Array1D<idg::Baseline>& baselines_ref)
{
    for (unsigned i = 0; i < NR_BASELINES; i++) {
        std::array<int, 2> baseline = stations[i];
        idg::Baseline baseline_ref  = baselines_ref(i);

        if ((abs(baseline[0] - (int) baseline_ref.station1) != 0) || (abs(baseline[1] - (int) baseline_ref.station2) != 0)) {
            std::cout << "This is station Baseline: " << i << std::endl;
        }
    }
}

void compare_spheroidal_buffer(
    std::array<float, SUBGRID_SIZE * SUBGRID_SIZE> spheroidal,
    idg::Array2D<float>& spheroidal_ref)
{
    for (unsigned i = 0; i < SUBGRID_SIZE; i++) {
        for (unsigned j = 0; j < SUBGRID_SIZE; j++) {
            float sph = spheroidal[i * SUBGRID_SIZE + j];
            float sph_ref = spheroidal_ref(i, j);

            if (abs(sph - sph_ref) != 0) {
                std::cout << "This is spheroidal Position: " << i << std::endl;
            }
        }
    }
}

void compare_aterms_buffer(
    std::array<std::array<std::complex<float>, 4>, NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE> aterms,
    idg::Array4D<idg::Matrix2x2<std::complex<float>>>& aterms_ref)
{
    for (unsigned ts = 0; ts < NR_TIMESLOTS; ts++) {
        for (unsigned station = 0; station < NR_STATIONS; station++) {
            for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
                for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
                    std::array<std::complex<float>, 4> term = aterms[ts * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE +
                                                                     station * SUBGRID_SIZE * SUBGRID_SIZE +
                                                                     y * SUBGRID_SIZE + x];
                    idg::Matrix2x2<std::complex<float>> term_ref = aterms_ref(ts, station, y, x);

                    if ((abs(term[0] - term_ref.xx) != 0) ||
                        (abs(term[1] - term_ref.xy) != 0) ||
                        (abs(term[2] - term_ref.yx) != 0) ||
                        (abs(term[3] - term_ref.yy) != 0)) {
                        std::cout << "This is aterms Timeslot, Station, Y, X: " << ts << " " << station << " " << y << " " << x << std::endl;
                    }
                }
            }
        }
    }
}

void compare_metadata_buffer(
    std::array<std::array<int, 9>, NR_BASELINES * NR_TIMESLOTS> metadata,
    idg::Array1D<idg::Metadata>& metadata_ref)
{
    for (int i = 0; i < NR_BASELINES; i++) {
        for (int j = 0; j < NR_TIMESLOTS; j++) {
            std::array<int, 9> m = metadata[i * NR_TIMESLOTS + j];
            idg::Metadata      m_ref = metadata_ref(i * NR_TIMESLOTS + j);

            if (abs(m[0] - m_ref.baseline_offset) != 0 ||
                abs(m[1] - m_ref.time_offset) != 0 ||
                abs(m[2] - m_ref.nr_timesteps) != 0 ||
                abs(m[3] - m_ref.aterm_index) != 0 ||
                abs(m[4] - (int) m_ref.baseline.station1) != 0 ||
                abs(m[5] - (int) m_ref.baseline.station2) != 0 ||
                abs(m[6] - m_ref.coordinate.x) != 0 ||
                abs(m[7] - m_ref.coordinate.y) != 0 ||
                abs(m[8] - m_ref.coordinate.z) != 0) {
                std::cout << "This is metadata Position: " << (i * NR_TIMESLOTS + j) << std::endl;
            }
        }
    }
}

void compare_subgrid_buffer(
    std::array<std::complex<float>, NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE> subgrids,
    idg::Array4D<std::complex<float>>& subgrids_ref)
{
    for (unsigned s = 0; s < NR_SUBGRIDS; s++) {
        for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
            for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
                for (unsigned c = 0; c < NR_CORRELATIONS; c++) {
                    std::complex<float> value = subgrids[s * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE +
                                                         c * SUBGRID_SIZE * SUBGRID_SIZE + y * SUBGRID_SIZE + x];
                    std::complex<float> value_ref = subgrids_ref(s, c, y, x);

                    if (abs(value - value_ref) != 0) {
                        std::cout << "This is subgrids Subgrid, Subgrid position (y, x), Correlation: " << s << " " << y << " " << x << " " << c << std::endl;
                        std::cout.precision(17);
                        std::cout << std::fixed << abs(value.real() - value_ref.real()) << " " << abs(value.imag() - value_ref.imag()) << std::endl;
                        std::cout << std::endl;
                    }
                }
            }
        }
    }
}

// --------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------

void compare_UVW_implicit(
    float *u_arr,
    float *v_arr,
    float *w_arr,
    idg::Array2D<idg::UVWCoordinate<float>>& uvw_ref)
{
    for (unsigned i = 0; i < NR_BASELINES; i++) {
        for (unsigned j = 0; j < NR_TIMESTEPS; j++) {
            float u = u_arr[i * NR_TIMESTEPS + j];
            float v = v_arr[i * NR_TIMESTEPS + j];
            float w = w_arr[i * NR_TIMESTEPS + j];

            float u_ref = uvw_ref(i, j).u;
            float v_ref = uvw_ref(i, j).v;
            float w_ref = uvw_ref(i, j).w;

            if ((abs(u - u_ref) != 0) || (abs(v - v_ref) != 0) || (abs(w - w_ref) != 0)) {
                std::cout << "This is uvw Baselines, Timestep: " << i << " " << j << std::endl;
            }
        }
    }
}

void compare_frequencies_implicit(
    double *frequencies,
    idg::Array1D<float>& frequencies_ref)
{
    for (unsigned i = 0; i < NR_CHANNELS; i++) {
        float freq = frequencies[i];
        float freq_ref = frequencies_ref(i);

        if (abs(freq - freq_ref) != 0) {
            std::cout << "This is frequencies Channel: " << i << std::endl;
        }
    }
}

void compare_wavenumbers_implicit(
    float *wavenumbers,
    idg::Array1D<float>& wavenumbers_ref)
{
    for (unsigned i = 0; i < NR_CHANNELS; i++) {
        float wav = wavenumbers[i];
        float wav_ref = wavenumbers_ref(i);

        if (abs(wav - wav_ref) != 0) {
            std::cout.precision(17);
            std::cout << "This is wavenumbers Channel: " << i << std::endl;
            std::cout << std::fixed << wav << " " << wav_ref << std::endl;
            std::cout << std::fixed << abs(wav - wav_ref) << std::endl;
            std::cout << std::endl;
        }
    }
}

void compare_visibilities_implicit(
    std::array<std::complex<float>, 4> *visibilities,
    idg::Array3D<idg::Visibility<std::complex<float>>>& visibilities_ref)
{
    int counter = 0;
    for (unsigned bl = 0; bl < NR_BASELINES; bl++) {
        for (unsigned time = 0; time < NR_TIMESTEPS; time++) {
            for (unsigned chan = 0; chan < NR_CHANNELS; chan++) {
                std::array<std::complex<float>, 4> vis       = visibilities[bl * NR_TIMESTEPS * NR_CHANNELS + time * NR_CHANNELS + chan];
                idg::Visibility<std::complex<float>> vis_ref = visibilities_ref(bl, time, chan);

                // NOTE: not exactly same which is strange...
                float tol = 0;

                if ((abs(vis[0] - vis_ref.xx) > tol) ||
                    (abs(vis[1] - vis_ref.xy) > tol) ||
                    (abs(vis[2] - vis_ref.yx) > tol) ||
                    (abs(vis[3] - vis_ref.yy) > tol)) {
                    std::cout << "This is visibilities Baseline, Timestep, Channel: " << bl << " " << time << " " << chan << std::endl;
                    std::cout.precision(17);
                    std::cout << std::fixed << abs(vis[0].real() - vis_ref.xx.real()) << " " << abs(vis[0].imag() - vis_ref.xx.imag()) << std::endl;
                    std::cout << std::fixed << abs(vis[1].real() - vis_ref.xy.real()) << " " << abs(vis[1].imag() - vis_ref.xy.imag()) << std::endl;
                    std::cout << std::fixed << abs(vis[2].real() - vis_ref.yx.real()) << " " << abs(vis[2].imag() - vis_ref.yx.imag()) << std::endl;
                    std::cout << std::fixed << abs(vis[3].real() - vis_ref.yy.real()) << " " << abs(vis[3].imag() - vis_ref.yy.imag()) << std::endl;

                    counter += 1;
                }
            }
        }
    }
    // std::cout << "Total not right: " << counter << std::endl;
    // std::cout << "Total length: " << (NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS) << std::endl;
}

void compare_baselines_implicit(
    std::array<int, 2> *stations,
    idg::Array1D<idg::Baseline>& baselines_ref)
{
    for (unsigned i = 0; i < NR_BASELINES; i++) {
        std::array<int, 2> baseline = stations[i];
        idg::Baseline baseline_ref  = baselines_ref(i);

        if ((abs(baseline[0] - (int) baseline_ref.station1) != 0) || (abs(baseline[1] - (int) baseline_ref.station2) != 0)) {
            std::cout << "This is station Baseline: " << i << std::endl;
        }
    }
}

void compare_spheroidal_implicit(
    float *spheroidal,
    idg::Array2D<float>& spheroidal_ref)
{
    for (unsigned i = 0; i < SUBGRID_SIZE; i++) {
        for (unsigned j = 0; j < SUBGRID_SIZE; j++) {
            float sph = spheroidal[i * SUBGRID_SIZE + j];
            float sph_ref = spheroidal_ref(i, j);

            if (abs(sph - sph_ref) != 0) {
                std::cout << "This is spheroidal Position: " << i << std::endl;
            }
        }
    }
}

void compare_aterms_implicit(
    std::array<std::complex<float>, 4> *aterms,
    idg::Array4D<idg::Matrix2x2<std::complex<float>>>& aterms_ref)
{
    for (unsigned ts = 0; ts < NR_TIMESLOTS; ts++) {
        for (unsigned station = 0; station < NR_STATIONS; station++) {
            for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
                for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
                    std::array<std::complex<float>, 4> term = aterms[ts * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE +
                                                                     station * SUBGRID_SIZE * SUBGRID_SIZE +
                                                                     y * SUBGRID_SIZE + x];
                    idg::Matrix2x2<std::complex<float>> term_ref = aterms_ref(ts, station, y, x);

                    if ((abs(term[0] - term_ref.xx) != 0) ||
                        (abs(term[1] - term_ref.xy) != 0) ||
                        (abs(term[2] - term_ref.yx) != 0) ||
                        (abs(term[3] - term_ref.yy) != 0)) {
                        std::cout << "This is aterms Timeslot, Station, Y, X: " << ts << " " << station << " " << y << " " << x << std::endl;
                    }
                }
            }
        }
    }
}

void compare_metadata_implicit(
    std::array<int, 9> *metadata,
    idg::Array1D<idg::Metadata>& metadata_ref)
{
    for (int i = 0; i < NR_BASELINES; i++) {
        for (int j = 0; j < NR_TIMESLOTS; j++) {
            std::array<int, 9> m = metadata[i * NR_TIMESLOTS + j];
            idg::Metadata      m_ref = metadata_ref(i * NR_TIMESLOTS + j);

            if (abs(m[0] - m_ref.baseline_offset) != 0 ||
                abs(m[1] - m_ref.time_offset) != 0 ||
                abs(m[2] - m_ref.nr_timesteps) != 0 ||
                abs(m[3] - m_ref.aterm_index) != 0 ||
                abs(m[4] - (int) m_ref.baseline.station1) != 0 ||
                abs(m[5] - (int) m_ref.baseline.station2) != 0 ||
                abs(m[6] - m_ref.coordinate.x) != 0 ||
                abs(m[7] - m_ref.coordinate.y) != 0 ||
                abs(m[8] - m_ref.coordinate.z) != 0) {
                std::cout << "This is metadata Position: " << (i * NR_TIMESLOTS + j) << std::endl;
            }
        }
    }
}

void compare_subgrid_implicit(
    std::complex<float> *subgrids,
    idg::Array4D<std::complex<float>>& subgrids_ref)
{
    for (unsigned s = 0; s < NR_SUBGRIDS; s++) {
        for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
            for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
                for (unsigned c = 0; c < NR_CORRELATIONS; c++) {
                    std::complex<float> value = subgrids[s * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE +
                                                         c * SUBGRID_SIZE * SUBGRID_SIZE + y * SUBGRID_SIZE + x];
                    std::complex<float> value_ref = subgrids_ref(s, c, y, x);

                    if (abs(value - value_ref) != 0) {
                        std::cout << "This is subgrids Subgrid, Subgrid position (y, x), Correlation: " << s << " " << y << " " << x << " " << c << std::endl;
                        std::cout.precision(17);
                        std::cout << std::fixed << abs(value.real() - value_ref.real()) << " " << abs(value.imag() - value_ref.imag()) << std::endl;
                        std::cout << std::endl;
                    }
                }
            }
        }
    }
}


// --------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------

void compare_UVW_explicit(
    float u_arr[NR_BASELINES * NR_TIMESTEPS],
    float v_arr[NR_BASELINES * NR_TIMESTEPS],
    float w_arr[NR_BASELINES * NR_TIMESTEPS],
    idg::Array2D<idg::UVWCoordinate<float>>& uvw_ref)
{
    for (unsigned i = 0; i < NR_BASELINES; i++) {
        for (unsigned j = 0; j < NR_TIMESTEPS; j++) {
            float u = u_arr[i * NR_TIMESTEPS + j];
            float v = v_arr[i * NR_TIMESTEPS + j];
            float w = w_arr[i * NR_TIMESTEPS + j];

            float u_ref = uvw_ref(i, j).u;
            float v_ref = uvw_ref(i, j).v;
            float w_ref = uvw_ref(i, j).w;

            if ((abs(u - u_ref) != 0) || (abs(v - v_ref) != 0) || (abs(w - w_ref) != 0)) {
                std::cout << "This is uvw Baselines, Timestep: " << i << " " << j << std::endl;
            }
        }
    }
}

void compare_frequencies_explicit(
    double frequencies[NR_CHANNELS],
    idg::Array1D<float>& frequencies_ref)
{
    for (unsigned i = 0; i < NR_CHANNELS; i++) {
        float freq = frequencies[i];
        float freq_ref = frequencies_ref(i);

        if (abs(freq - freq_ref) != 0) {
            std::cout << "This is frequencies Channel: " << i << std::endl;
        }
    }
}

void compare_wavenumbers_explicit(
    float wavenumbers[NR_CHANNELS],
    idg::Array1D<float>& wavenumbers_ref)
{
    for (unsigned i = 0; i < NR_CHANNELS; i++) {
        float wav = wavenumbers[i];
        float wav_ref = wavenumbers_ref(i);

        if (abs(wav - wav_ref) != 0) {
            std::cout.precision(17);
            std::cout << "This is wavenumbers Channel: " << i << std::endl;
            std::cout << std::fixed << wav << " " << wav_ref << std::endl;
            std::cout << std::fixed << abs(wav - wav_ref) << std::endl;
            std::cout << std::endl;
        }
    }
}

void compare_visibilities_explicit(
    std::array<std::complex<float>, 4> *visibilities,
    idg::Array3D<idg::Visibility<std::complex<float>>>& visibilities_ref)
{
    int counter = 0;
    for (unsigned bl = 0; bl < NR_BASELINES; bl++) {
        for (unsigned time = 0; time < NR_TIMESTEPS; time++) {
            for (unsigned chan = 0; chan < NR_CHANNELS; chan++) {
                std::array<std::complex<float>, 4> vis       = visibilities[bl * NR_TIMESTEPS * NR_CHANNELS + time * NR_CHANNELS + chan];
                idg::Visibility<std::complex<float>> vis_ref = visibilities_ref(bl, time, chan);

                // NOTE: not exactly same which is strange...
                float tol = 0;

                if ((abs(vis[0] - vis_ref.xx) > tol) ||
                    (abs(vis[1] - vis_ref.xy) > tol) ||
                    (abs(vis[2] - vis_ref.yx) > tol) ||
                    (abs(vis[3] - vis_ref.yy) > tol)) {
                    std::cout << "This is visibilities Baseline, Timestep, Channel: " << bl << " " << time << " " << chan << std::endl;
                    std::cout.precision(17);
                    std::cout << std::fixed << abs(vis[0].real() - vis_ref.xx.real()) << " " << abs(vis[0].imag() - vis_ref.xx.imag()) << std::endl;
                    std::cout << std::fixed << abs(vis[1].real() - vis_ref.xy.real()) << " " << abs(vis[1].imag() - vis_ref.xy.imag()) << std::endl;
                    std::cout << std::fixed << abs(vis[2].real() - vis_ref.yx.real()) << " " << abs(vis[2].imag() - vis_ref.yx.imag()) << std::endl;
                    std::cout << std::fixed << abs(vis[3].real() - vis_ref.yy.real()) << " " << abs(vis[3].imag() - vis_ref.yy.imag()) << std::endl;

                    counter += 1;
                }
            }
        }
    }
    // std::cout << "Total not right: " << counter << std::endl;
    // std::cout << "Total length: " << (NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS) << std::endl;
}

void compare_baselines_explicit(
    std::array<int, 2> stations[NR_BASELINES],
    idg::Array1D<idg::Baseline>& baselines_ref)
{
    for (unsigned i = 0; i < NR_BASELINES; i++) {
        std::array<int, 2> baseline = stations[i];
        idg::Baseline baseline_ref  = baselines_ref(i);

        if ((abs(baseline[0] - (int) baseline_ref.station1) != 0) || (abs(baseline[1] - (int) baseline_ref.station2) != 0)) {
            std::cout << "This is station Baseline: " << i << std::endl;
        }
    }
}

void compare_spheroidal_explicit(
    float spheroidal[SUBGRID_SIZE * SUBGRID_SIZE],
    idg::Array2D<float>& spheroidal_ref)
{
    for (unsigned i = 0; i < SUBGRID_SIZE; i++) {
        for (unsigned j = 0; j < SUBGRID_SIZE; j++) {
            float sph = spheroidal[i * SUBGRID_SIZE + j];
            float sph_ref = spheroidal_ref(i, j);

            if (abs(sph - sph_ref) != 0) {
                std::cout << "This is spheroidal Position: " << i << std::endl;
            }
        }
    }
}

void compare_aterms_explicit(
    std::array<std::complex<float>, 4> aterms[NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE],
    idg::Array4D<idg::Matrix2x2<std::complex<float>>>& aterms_ref)
{
    for (unsigned ts = 0; ts < NR_TIMESLOTS; ts++) {
        for (unsigned station = 0; station < NR_STATIONS; station++) {
            for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
                for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
                    std::array<std::complex<float>, 4> term = aterms[ts * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE +
                                                                     station * SUBGRID_SIZE * SUBGRID_SIZE +
                                                                     y * SUBGRID_SIZE + x];
                    idg::Matrix2x2<std::complex<float>> term_ref = aterms_ref(ts, station, y, x);

                    if ((abs(term[0] - term_ref.xx) != 0) ||
                        (abs(term[1] - term_ref.xy) != 0) ||
                        (abs(term[2] - term_ref.yx) != 0) ||
                        (abs(term[3] - term_ref.yy) != 0)) {
                        std::cout << "This is aterms Timeslot, Station, Y, X: " << ts << " " << station << " " << y << " " << x << std::endl;
                    }
                }
            }
        }
    }
}

void compare_metadata_explicit(
    std::array<int, 9> metadata[NR_BASELINES * NR_TIMESLOTS],
    idg::Array1D<idg::Metadata>& metadata_ref)
{
    for (int i = 0; i < NR_BASELINES; i++) {
        for (int j = 0; j < NR_TIMESLOTS; j++) {
            std::array<int, 9> m = metadata[i * NR_TIMESLOTS + j];
            idg::Metadata      m_ref = metadata_ref(i * NR_TIMESLOTS + j);

            if (abs(m[0] - m_ref.baseline_offset) != 0 ||
                abs(m[1] - m_ref.time_offset) != 0 ||
                abs(m[2] - m_ref.nr_timesteps) != 0 ||
                abs(m[3] - m_ref.aterm_index) != 0 ||
                abs(m[4] - (int) m_ref.baseline.station1) != 0 ||
                abs(m[5] - (int) m_ref.baseline.station2) != 0 ||
                abs(m[6] - m_ref.coordinate.x) != 0 ||
                abs(m[7] - m_ref.coordinate.y) != 0 ||
                abs(m[8] - m_ref.coordinate.z) != 0) {
                std::cout << "This is metadata Position: " << (i * NR_TIMESLOTS + j) << std::endl;
            }
        }
    }
}

void compare_subgrid_explicit(
    std::array<std::complex<float>, NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE> subgrid,
    idg::Array4D<std::complex<float>>& subgrids_ref)
{
    for (unsigned s = 0; s < NR_SUBGRIDS; s++) {
        for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
            for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
                for (unsigned c = 0; c < NR_CORRELATIONS; c++) {
                    std::complex<float> value = subgrid[s * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE +
                                                         c * SUBGRID_SIZE * SUBGRID_SIZE + y * SUBGRID_SIZE + x];
                    std::complex<float> value_ref = subgrids_ref(s, c, y, x);

                    if (abs(value - value_ref) != 0) {
                        std::cout << "This is subgrids Subgrid, Subgrid position (y, x), Correlation: " << s << " " << y << " " << x << " " << c << std::endl;
                        std::cout.precision(17);
                        std::cout << std::fixed << abs(value.real() - value_ref.real()) << " " << abs(value.imag() - value_ref.imag()) << std::endl;
                        std::cout << std::endl;
                    }
                }
            }
        }
    }
}

// --------------------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------------------

// computes sqrt(A^2 - B^2) / n
void get_accuracy_buffer(
    // std::array<std::complex<float>, NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE> subgrids,
    std::vector<std::complex<float>>& subgrids,
    idg::Array4D<std::complex<float>>& subgrids_ref)
{
    double r_error = 0.0;
    double i_error = 0.0;
    int nnz = 0;

    float r_max = 1;
    float i_max = 1;

    for (unsigned s = 0; s < NR_SUBGRIDS; s++) {
        for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
            for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
                for (unsigned c = 0; c < NR_CORRELATIONS; c++) {
                    float r_value = abs(subgrids_ref(s, c, y, x).real());
                    float i_value = abs(subgrids_ref(s, c, y, x).imag());

                    if (r_value > r_max) {
                        r_max = r_value;
                    }
                    if (i_value > i_max) {
                        i_max = i_value;
                    }
                }
            }
        }
    }

    int nerrors = 0;

    for (unsigned s = 0; s < NR_SUBGRIDS; s++) {
        for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
            for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
                for (unsigned c = 0; c < NR_CORRELATIONS; c++) {
                    int index = s * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE + c * SUBGRID_SIZE * SUBGRID_SIZE + y * SUBGRID_SIZE + x;

                    float r_cmp = subgrids[index].real();
                    float i_cmp = subgrids[index].imag();
                    float r_ref = subgrids_ref(s, c, y, x).real();
                    float i_ref = subgrids_ref(s, c, y, x).imag();

                    double r_diff = r_ref - r_cmp;
                    double i_diff = i_ref - i_cmp;

                    if (abs(subgrids_ref(s, c, y, x)) > 0.0f) {
                        if ((abs(r_diff) > 0.0f || abs(i_diff) > 0.0f) && nerrors < 16) {
                            // printf("(%f, %f) - (%f, %f) = (%f, %f)\n", r_cmp, i_cmp, r_ref, i_ref, r_diff, i_diff);
                            nerrors++;
                        }
                        nnz++;
                        r_error += (r_diff * r_diff) / r_max;
                        i_error += (i_diff * i_diff) / i_max;
                    }
                }
            }
        }
    }
    printf("r_error: %f\n", r_error);
    printf("i_error: %f\n", i_error);
    printf("r_max: %f\n", r_max);
    printf("i_max: %f\n", i_max);
    printf("nnz: %d\n", nnz);

    r_error /= fmax(1, nnz);
    i_error /= fmax(1, nnz);

    float grid_error = sqrt(r_error + i_error);

    float tol = GRID_SIZE * GRID_SIZE * std::numeric_limits<float>::epsilon();
    if (grid_error < tol) {
        std::cout << "Gridding test PASSED!" << std::endl;
    } else {
        std::cout << "Gridding test FAILED!" << std::endl;
    }

    std::cout << "grid_error = " << std::scientific << grid_error << std::endl;
}


// computes sqrt(A^2 - B^2) / n
void get_accuracy_implicit(
    std::complex<float> *subgrids,
    idg::Array4D<std::complex<float>>& subgrids_ref)
{
    double r_error = 0.0;
    double i_error = 0.0;
    int nnz = 0;

    float r_max = 1;
    float i_max = 1;

    for (unsigned s = 0; s < NR_SUBGRIDS; s++) {
        for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
            for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
                for (unsigned c = 0; c < NR_CORRELATIONS; c++) {
                    float r_value = abs(subgrids_ref(s, c, y, x).real());
                    float i_value = abs(subgrids_ref(s, c, y, x).imag());

                    if (r_value > r_max) {
                        r_max = r_value;
                    }
                    if (i_value > i_max) {
                        i_max = i_value;
                    }
                }
            }
        }
    }

    int nerrors = 0;

    for (unsigned s = 0; s < NR_SUBGRIDS; s++) {
        for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
            for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
                for (unsigned c = 0; c < NR_CORRELATIONS; c++) {
                    int index = s * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE + c * SUBGRID_SIZE * SUBGRID_SIZE + y * SUBGRID_SIZE + x;

                    float r_cmp = subgrids[index].real();
                    float i_cmp = subgrids[index].imag();
                    float r_ref = subgrids_ref(s, c, y, x).real();
                    float i_ref = subgrids_ref(s, c, y, x).imag();

                    double r_diff = r_ref - r_cmp;
                    double i_diff = i_ref - i_cmp;

                    if (abs(subgrids_ref(s, c, y, x)) > 0.0f) {
                        if ((abs(r_diff) > 0.0f || abs(i_diff) > 0.0f) && nerrors < 16) {
                            // printf("(%f, %f) - (%f, %f) = (%f, %f)\n", r_cmp, i_cmp, r_ref, i_ref, r_diff, i_diff);
                            nerrors++;
                        }
                        nnz++;
                        r_error += (r_diff * r_diff) / r_max;
                        i_error += (i_diff * i_diff) / i_max;
                    }
                }
            }
        }
    }
    printf("r_error: %f\n", r_error);
    printf("i_error: %f\n", i_error);
    printf("r_max: %f\n", r_max);
    printf("i_max: %f\n", i_max);
    printf("nnz: %d\n", nnz);

    r_error /= fmax(1, nnz);
    i_error /= fmax(1, nnz);

    float grid_error = sqrt(r_error + i_error);

    float tol = GRID_SIZE * GRID_SIZE * std::numeric_limits<float>::epsilon();
    if (grid_error < tol) {
        std::cout << "Gridding test PASSED!" << std::endl;
    } else {
        std::cout << "Gridding test FAILED!" << std::endl;
    }

    std::cout << "grid_error = " << std::scientific << grid_error << std::endl;
}



// computes sqrt(A^2 - B^2) / n
void get_accuracy_explicit(
    std::array<std::complex<float>, NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE> subgrid,
    idg::Array4D<std::complex<float>>& subgrids_ref)
{
    double r_error = 0.0;
    double i_error = 0.0;
    int nnz = 0;

    float r_max = 1;
    float i_max = 1;

    for (unsigned s = 0; s < NR_SUBGRIDS; s++) {
        for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
            for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
                for (unsigned c = 0; c < NR_CORRELATIONS; c++) {
                    float r_value = abs(subgrids_ref(s, c, y, x).real());
                    float i_value = abs(subgrids_ref(s, c, y, x).imag());

                    if (r_value > r_max) {
                        r_max = r_value;
                    }
                    if (i_value > i_max) {
                        i_max = i_value;
                    }
                }
            }
        }
    }

    int nerrors = 0;

    for (unsigned s = 0; s < NR_SUBGRIDS; s++) {
        for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
            for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
                for (unsigned c = 0; c < NR_CORRELATIONS; c++) {
                    int index = s * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE + c * SUBGRID_SIZE * SUBGRID_SIZE + y * SUBGRID_SIZE + x;

                    float r_cmp = subgrid[index].real();
                    float i_cmp = subgrid[index].imag();
                    float r_ref = subgrids_ref(s, c, y, x).real();
                    float i_ref = subgrids_ref(s, c, y, x).imag();

                    double r_diff = r_ref - r_cmp;
                    double i_diff = i_ref - i_cmp;

                    if (abs(subgrids_ref(s, c, y, x)) > 0.0f) {
                        if ((abs(r_diff) > 0.0f || abs(i_diff) > 0.0f) && nerrors < 16) {
                            // printf("(%f, %f) - (%f, %f) = (%f, %f)\n", r_cmp, i_cmp, r_ref, i_ref, r_diff, i_diff);
                            nerrors++;
                        }
                        nnz++;
                        r_error += (r_diff * r_diff) / r_max;
                        i_error += (i_diff * i_diff) / i_max;
                    }
                }
            }
        }
    }
    printf("r_error: %f\n", r_error);
    printf("i_error: %f\n", i_error);
    printf("r_max: %f\n", r_max);
    printf("i_max: %f\n", i_max);
    printf("nnz: %d\n", nnz);

    r_error /= fmax(1, nnz);
    i_error /= fmax(1, nnz);

    float grid_error = sqrt(r_error + i_error);

    float tol = GRID_SIZE * GRID_SIZE * std::numeric_limits<float>::epsilon();
    if (grid_error < tol) {
        std::cout << "Gridding test PASSED!" << std::endl;
    } else {
        std::cout << "Gridding test FAILED!" << std::endl;
    }

    std::cout << "grid_error = " << std::scientific << grid_error << std::endl;
}
