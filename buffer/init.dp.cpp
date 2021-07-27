#include "init.h"


void initialize_uvw(
    std::array<float, NR_BASELINES * NR_TIMESTEPS>& u_arr,
    std::array<float, NR_BASELINES * NR_TIMESTEPS>& v_arr,
    std::array<float, NR_BASELINES * NR_TIMESTEPS>& w_arr)
{
    for (unsigned bl = 0; bl < NR_BASELINES; bl++) {
        // Get random radius
        float radius_u = (GRID_SIZE / 2) + (double) rand() / (double) (RAND_MAX) * (GRID_SIZE / 2);
        float radius_v = (GRID_SIZE / 2) + (double) rand() / (double) (RAND_MAX) * (GRID_SIZE / 2);

        // Evaluate elipsoid
        for (unsigned t = 0; t < NR_TIMESTEPS; t++) {
            float angle = (t + 0.5) / (360.0f / NR_TIMESTEPS);
            float u = radius_u * cos(angle * M_PI);
            float v = radius_v * sin(angle * M_PI);
            float w = 0;

            u_arr[bl * NR_TIMESTEPS + t] = u;
            v_arr[bl * NR_TIMESTEPS + t] = v;
            w_arr[bl * NR_TIMESTEPS + t] = w;
        }
    }
}

void initialize_frequencies(
    std::array<double, NR_CHANNELS>& frequencies)
{
    const unsigned int start_frequency = 150e6;
    const float frequency_increment = 0.7e6;
    for (unsigned i = 0; i < NR_CHANNELS; i++) {
        double frequency = start_frequency + frequency_increment * i;
        frequencies[i] = frequency;
    }
}

void initialize_wavenumbers(
    const std::array<double, NR_CHANNELS> frequencies,
    std::array<float, NR_CHANNELS>& wavenumbers)
{
    const double speed_of_light = 299792458.0;
    for (unsigned i = 0; i < NR_CHANNELS; i++) {
        wavenumbers[i] = 2 * M_PI * frequencies[i] / speed_of_light;
    }
}

void initialize_visibilities(
    const std::array<double, NR_CHANNELS> frequencies,
    const std::array<float, NR_BASELINES * NR_TIMESTEPS> uCoor,
    const std::array<float, NR_BASELINES * NR_TIMESTEPS> vCoor,
    std::vector<std::array<std::complex<float>, 4>>& visibilities)
{
    float x_offset = 0.6 * GRID_SIZE;
    float y_offset = 0.7 * GRID_SIZE;
    float amplitude = 1.0f;
    float l = x_offset * IMAGE_SIZE / GRID_SIZE;
    float m = y_offset * IMAGE_SIZE / GRID_SIZE;

    for (unsigned bl = 0; bl < NR_BASELINES; bl++) {
        for (unsigned time = 0; time < NR_TIMESTEPS; time++) {
            for (unsigned chan = 0; chan < NR_CHANNELS; chan++) {
                const double speed_of_light = 299792458.0;
                float u = (frequencies[chan] / speed_of_light) * uCoor[bl * NR_TIMESTEPS + time];
                float v = (frequencies[chan] / speed_of_light) * vCoor[bl * NR_TIMESTEPS + time];

                std::complex<float> value = amplitude * exp(std::complex<float>(0 , -2 * M_PI * (u * l + v * m)));

                std::complex<float> xx(value * 1.01f);
                std::complex<float> xy(value * 1.02f);
                std::complex<float> yx(value * 1.03f);
                std::complex<float> yy(value * 1.04f);

                visibilities[bl * NR_TIMESTEPS * NR_CHANNELS + time * NR_CHANNELS + chan] = {xx, xy, yx, yy};
            }
        }
    }
}

void initialize_baselines(
    std::array<std::array<int, 2>, NR_BASELINES>& stations)
{
    unsigned bl = 0;
    for (unsigned station1 = 0 ; station1 < NR_STATIONS; station1++) {
        for (unsigned station2 = station1 + 1; station2 < NR_STATIONS; station2++) {
            if (bl >= NR_BASELINES) {
                break;
            }
            stations[bl][0] = station1;
            stations[bl][1] = station2;
            bl++;
        }
    }
}

void initialize_spheroidal(
    std::array<float, SUBGRID_SIZE * SUBGRID_SIZE>& spheroidal)
{
    for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
        float tmp_y = fabs(-1 + y * 2.0f / float(SUBGRID_SIZE));
        for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
            float tmp_x = fabs(-1 + x * 2.0f / float(SUBGRID_SIZE));
            spheroidal[y * SUBGRID_SIZE + x] = tmp_y * tmp_x;
        }
    }
}

void initialize_aterms(
    const std::array<float, SUBGRID_SIZE * SUBGRID_SIZE> spheroidal,
    std::array<std::array<std::complex<float>, 4>, NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE>& aterms)
{
    for (unsigned ts = 0; ts < NR_TIMESLOTS; ts++) {
        for (unsigned station = 0; station < NR_STATIONS; station++) {
            for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
                for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
                    float scale = 0.8 + ((double) rand() / (double) (RAND_MAX) * 0.4);
                    float value = spheroidal[y * SUBGRID_SIZE + x] * scale;

                    std::complex<float> xx = std::complex<float>(value + 0.1, -0.1);
                    std::complex<float> xy = std::complex<float>(value - 0.2,  0.1);
                    std::complex<float> yx = std::complex<float>(value - 0.2,  0.1);
                    std::complex<float> yy = std::complex<float>(value + 0.1, -0.1);

                    aterms[ts * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE + station * SUBGRID_SIZE * SUBGRID_SIZE +
                           y * SUBGRID_SIZE + x] = {xx, xy, yx, yy};
                }
            }
        }
    }
}

void initialize_metadata(
    const std::array<std::array<int, 2>, NR_BASELINES> stations,
    std::array<std::array<int, 9>, NR_BASELINES * NR_TIMESLOTS>& metadata)
{
    for (auto bl = 0; bl < NR_BASELINES; bl++) {
        for (auto ts = 0; ts < NR_TIMESLOTS; ts++) {
            // Metadata settings
            int baseline_offset = 0;
            int time_offset = bl * NR_TIMESLOTS * NR_TIMESTEPS_SUBGRID +
                                             ts * NR_TIMESTEPS_SUBGRID;
            int aterm_index = 0; // use the same aterm for every timeslot

            int x = (double) rand() / (double) (RAND_MAX) * GRID_SIZE;
            int y = (double) rand() / (double) (RAND_MAX) * GRID_SIZE;
            // NOTE: original code a z coordinate is part of the Coordinate struct but never initialized, so I
            // assumed it is just always zero.
            int z = 0;

            // Set metadata for current subgrid
            std::array<int, 9> m = {baseline_offset, time_offset, (int) NR_TIMESTEPS_SUBGRID, aterm_index,
                                    stations[bl][0], stations[bl][1], x, y, z};
            metadata[bl * NR_TIMESLOTS + ts] = m;
        }
    }
}

void initialize_subgrids(
    std::array<std::complex<float>, NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE>& subgrid)
{
	// Initialize subgrids
	for (unsigned s = 0; s < NR_SUBGRIDS; s++) {
        for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
            for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
                for (unsigned c = 0; c < NR_CORRELATIONS; c++) {
	                std::complex<float> pixel_value(
						((y * SUBGRID_SIZE + x + 1) / ((float) 100 * SUBGRID_SIZE * SUBGRID_SIZE)),
						(c / 10.0f));

                    subgrid[s * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE + c * SUBGRID_SIZE * SUBGRID_SIZE +
                            y * SUBGRID_SIZE + x] = pixel_value;
	            }
	        }
	    }
	}
}
