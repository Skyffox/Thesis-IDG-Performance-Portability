#include "init.h"


void printUVW(
    const std::array<float, NR_BASELINES * NR_TIMESTEPS> u,
    const std::array<float, NR_BASELINES * NR_TIMESTEPS> v,
    const std::array<float, NR_BASELINES * NR_TIMESTEPS> w)
{
    std::cout << ">>> Printing all UVW coordinates." << std::endl;
    for (unsigned i = 0; i < NR_BASELINES; i++) {
        for (unsigned j = 0; j < NR_TIMESTEPS; j++) {
            std::cout << "This is Baseline, Timestep: " << i << ", " << j << std::endl;
            std::cout << "U coordinate: " << u[i * NR_TIMESTEPS + j] << std::endl;
            std::cout << "V coordinate: " << v[i * NR_TIMESTEPS + j] << std::endl;
            std::cout << "W coordinate: " << w[i * NR_TIMESTEPS + j] << std::endl;
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "Total length: " << (NR_BASELINES * NR_TIMESTEPS) << std::endl;
    std::cout << std::endl;
}

void printFrequencies(
    const std::array<double, NR_CHANNELS> frequencies)
{
    std::cout << ">>> Printing all Frequencies." << std::endl;
    for (unsigned i = 0; i < NR_CHANNELS; i++) {
        std::cout << "This is Channel: " << i << std::endl;
        std::cout << "Frequency: " << frequencies[i] << std::endl;
    }
    std::cout << "Total length: " << NR_CHANNELS << std::endl;
    std::cout << std::endl;
}

void printWavenumbers(
    const std::array<float, NR_CHANNELS> wavenumbers)
{
    std::cout << ">>> Printing all Wavenumbers." << std::endl;
    for (unsigned i = 0; i < NR_CHANNELS; i++) {
        std::cout << "This is Channel: " << i << std::endl;
        std::cout << "Wavenumber: " << wavenumbers[i] << std::endl;
    }
    std::cout << "Total length: " << NR_CHANNELS << std::endl;
    std::cout << std::endl;
}

void printVisibilities(
    const std::vector<std::array<std::complex<float>, 4>> visibilities)
{
    std::cout << ">>> Printing all Visibilities." << std::endl;
    for (unsigned bl = 0; bl < NR_BASELINES; bl++) {
        for (unsigned time = 0; time < NR_TIMESTEPS; time++) {
            for (unsigned chan = 0; chan < NR_CHANNELS; chan++) {
                std::array<std::complex<float>, 4> value = visibilities[bl * NR_TIMESTEPS * NR_CHANNELS +
                                                                        time * NR_CHANNELS + chan];

                std::cout << "This is Baseline, Timestep, Channel: " << bl << ", " << time << ", " << chan << std::endl;
                std::cout << "- XX: " << value[0] << std::endl;
                std::cout << "- XY: " << value[1] << std::endl;
                std::cout << "- YX: " << value[2] << std::endl;
                std::cout << "- YY: " << value[3] << std::endl;
                std::cout << std::endl;
            }
        }
    }
    std::cout << "Total length: " << (NR_BASELINES * NR_TIMESTEPS * NR_CHANNELS) << std::endl;
    std::cout << std::endl;
}

void printBaselines(
    const std::array<std::array<int, 2>, NR_BASELINES> stations)
{
    std::cout << ">>> Printing all Baselines." << std::endl;
    for (unsigned i = 0; i < NR_BASELINES; i++) {
        std::cout << "This is Baseline: " << i << std::endl;
        std::cout << "Station 1: " << stations[i][0] << std::endl;
        std::cout << "Station 2: " << stations[i][1] << std::endl;
    }
    std::cout << "Total length: " << NR_BASELINES << std::endl;
    std::cout << std::endl;
}

void printSpheroidal(
    const std::array<float, SUBGRID_SIZE * SUBGRID_SIZE> spheroidal)
{
    std::cout << ">>> Printing all Spheroidal." << std::endl;
    for (unsigned i = 0; i < SUBGRID_SIZE * SUBGRID_SIZE; i++) {
        std::cout << "This is Iteration: " << i << std::endl;
        std::cout << "Spheroidal: " << spheroidal[i] << std::endl;
    }
    std::cout << "Total length: " << (SUBGRID_SIZE * SUBGRID_SIZE) << std::endl;
    std::cout << std::endl;
}

void printAterms(
    const std::array<std::array<std::complex<float>, 4>, NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE> aterms)
{
    std::cout << ">>> Printing all Aterms." << std::endl;
    for (unsigned ts = 0; ts < NR_TIMESLOTS; ts++) {
        for (unsigned station = 0; station < NR_STATIONS; station++) {
            for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
                for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
                    std::array<std::complex<float>, 4> values = aterms[ts * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE +
                                                                       station * SUBGRID_SIZE * SUBGRID_SIZE +
                                                                       y * SUBGRID_SIZE + x];

                    std::cout << "This is Timeslot, Station, Subgrid position (y, x): " << ts << ", " << station << ", " << y << ", " << x << std::endl;
                    std::cout << "- XX: " << values[0] << std::endl;
                    std::cout << "- XY: " << values[1] << std::endl;
                    std::cout << "- YX: " << values[2] << std::endl;
                    std::cout << "- YY: " << values[3] << std::endl;
                    std::cout << std::endl;
                }
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::cout << "Total length: " << (NR_TIMESLOTS * NR_STATIONS * SUBGRID_SIZE * SUBGRID_SIZE) << std::endl;
    std::cout << std::endl;
}

void printMetadata(
    std::array<std::array<int, 9>, NR_BASELINES * NR_TIMESLOTS> metadata)
{
    std::cout << ">>> Printing all Metadata." << std::endl;
    for (int i = 0; i < NR_BASELINES; i++) {
        for (int j = 0; j < NR_TIMESLOTS; j++) {
            std::cout << "This is Baseline, Timeslot: " << i << ", " << j << std::endl;

            std::cout << "baseline offset: " << metadata[i * NR_TIMESLOTS + j][0] << std::endl;
            std::cout << "time offset:     " << metadata[i * NR_TIMESLOTS + j][1] << std::endl;
            std::cout << "nr timesteps:    " << metadata[i * NR_TIMESLOTS + j][2] << std::endl;
            std::cout << "a term index:    " << metadata[i * NR_TIMESLOTS + j][3] << std::endl;
            std::cout << "station1:        " << metadata[i * NR_TIMESLOTS + j][4] << std::endl;
            std::cout << "station2:        " << metadata[i * NR_TIMESLOTS + j][5] << std::endl;
            std::cout << "xcoordinate:     " << metadata[i * NR_TIMESLOTS + j][6] << std::endl;
            std::cout << "ycoordinate:     " << metadata[i * NR_TIMESLOTS + j][7] << std::endl;
            std::cout << "zcoordinate:     " << metadata[i * NR_TIMESLOTS + j][8] << std::endl;
            std::cout << std::endl;
        }
    }
    std::cout << "Total length: " << (NR_BASELINES * NR_TIMESLOTS) << std::endl;
    std::cout << std::endl;
}

void printSubgrid(
    std::array<std::complex<float>, NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE> subgrids)
{
    for (unsigned s = 0; s < NR_SUBGRIDS; s++) {
        for (unsigned y = 0; y < SUBGRID_SIZE; y++) {
            for (unsigned x = 0; x < SUBGRID_SIZE; x++) {
                for (unsigned c = 0; c < NR_CORRELATIONS; c++) {
                    std::complex<float> value = subgrids[s * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE +
                                                         c * SUBGRID_SIZE * SUBGRID_SIZE + y * SUBGRID_SIZE + x];
                    std::cout << "This is Subgrid, Subgrid position (y, x), Correlation: " << s << ", " << y << ", " << x << ", " << c << std::endl;
                    std::cout << value << std::endl;
                    std::cout << std::endl;
                }
            }
        }
    }
    std::cout << "Total length: " << (NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE) << std::endl;
    std::cout << std::endl;
}
