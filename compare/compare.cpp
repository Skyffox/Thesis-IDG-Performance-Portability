#include <iostream>
#include <fstream>
#include <complex>
#include <vector>


#define GRID_SIZE            1024
#define NR_CORRELATIONS      4
#define SUBGRID_SIZE         32
#define NR_STATIONS          10
#define NR_TIMESLOTS         2
#define NR_BASELINES         ((NR_STATIONS * (NR_STATIONS - 1)) / 2)
#define NR_SUBGRIDS          (NR_BASELINES * NR_TIMESLOTS)
#define ARRAY_SIZE           NR_SUBGRIDS * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE


// computes sqrt(A^2 - B^2) / n
void get_accuracy(
    std::vector<std::complex<float>> subgrid,
    std::vector<std::complex<float>> subgrid_ref)
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
                    int index = s * NR_CORRELATIONS * SUBGRID_SIZE * SUBGRID_SIZE + c * SUBGRID_SIZE * SUBGRID_SIZE + y * SUBGRID_SIZE + x;

                    float r_value = subgrid_ref[index].real();
                    float i_value = subgrid_ref[index].imag();

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
                    float r_ref = subgrid_ref[index].real();
                    float i_ref = subgrid_ref[index].imag();

                    double r_diff = r_ref - r_cmp;
                    double i_diff = i_ref - i_cmp;

                    if (fabs(subgrid_ref[index]) > 0.0f) {
                        if ((fabs(r_diff) > 0.0f || fabs(i_diff) > 0.0f) && nerrors < 16) {
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
    std::cout << std::endl;
}


void read_data(std::string name, std::vector<std::complex<float>>& arr) {
    std::string s = "/home/julius/Downloads/idg-fpga-master/thesis/compare/output/";

    std::ifstream file(s += name);
    int i = 0;
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::complex<float> c;
            std::istringstream is(line);
            is >> c;
            arr[i] = c;
            i++;
        }
        file.close();
    }
}


int main(int argc, char **argv) {
    // NOTE: ultimate reference
    std::vector<std::complex<float>> reference_i5(ARRAY_SIZE);

    std::vector<std::complex<float>> reference_gold(ARRAY_SIZE);
    std::vector<std::complex<float>> reference_platinum(ARRAY_SIZE);
    std::vector<std::complex<float>> reference_gen9(ARRAY_SIZE);
    std::vector<std::complex<float>> reference_gen9cpu(ARRAY_SIZE);
    std::vector<std::complex<float>> reference_iris(ARRAY_SIZE);
    std::vector<std::complex<float>> reference_iriscpu(ARRAY_SIZE);

    std::vector<std::complex<float>> buffer_i5(ARRAY_SIZE);
    std::vector<std::complex<float>> buffer_gold(ARRAY_SIZE);
    std::vector<std::complex<float>> buffer_platinum(ARRAY_SIZE);
    std::vector<std::complex<float>> buffer_gen9(ARRAY_SIZE);
    std::vector<std::complex<float>> buffer_gen9cpu(ARRAY_SIZE);
    std::vector<std::complex<float>> buffer_iris(ARRAY_SIZE);
    std::vector<std::complex<float>> buffer_iriscpu(ARRAY_SIZE);

    std::vector<std::complex<float>> implicit_i5(ARRAY_SIZE);
    std::vector<std::complex<float>> implicit_gold(ARRAY_SIZE);
    std::vector<std::complex<float>> implicit_platinum(ARRAY_SIZE);
    std::vector<std::complex<float>> implicit_gen9(ARRAY_SIZE);
    std::vector<std::complex<float>> implicit_gen9cpu(ARRAY_SIZE);
    std::vector<std::complex<float>> implicit_iris(ARRAY_SIZE);
    std::vector<std::complex<float>> implicit_iriscpu(ARRAY_SIZE);

    // -----------------------------------------------------------------------------------------------------------------

    read_data("reference/i5.txt", reference_i5);
    read_data("reference/gold.txt", reference_gold);
    read_data("reference/platinum.txt", reference_platinum);
    read_data("reference/gen9.txt", reference_gen9);
    read_data("reference/gen9cpu.txt", reference_gen9cpu);
    read_data("reference/iris.txt", reference_iris);
    read_data("reference/iriscpu.txt", reference_iriscpu);

    read_data("buffer/i5.txt", buffer_i5);
    read_data("buffer/gold.txt", buffer_gold);
    read_data("buffer/platinum.txt", buffer_platinum);
    read_data("buffer/gen9.txt", buffer_gen9);
    read_data("buffer/gen9cpu.txt", buffer_gen9cpu);
    read_data("buffer/iris.txt", buffer_iris);
    read_data("buffer/iriscpu.txt", buffer_iriscpu);

    read_data("implicit/i5.txt", implicit_i5);
    read_data("implicit/gold.txt", implicit_gold);
    read_data("implicit/platinum.txt", implicit_platinum);
    read_data("implicit/gen9.txt", implicit_gen9);
    read_data("implicit/gen9cpu.txt", implicit_gen9cpu);
    read_data("implicit/iris.txt", implicit_iris);
    read_data("implicit/iriscpu.txt", implicit_iriscpu);

    // -----------------------------------------------------------------------------------------------------------------

    std::cout << "Compare gold reference with i5 reference subgrids:" << std::endl;
    get_accuracy(reference_gold, reference_i5);

    std::cout << "Compare platinum reference with i5 reference subgrids:" << std::endl;
    get_accuracy(reference_platinum, reference_i5);

    std::cout << "Compare gen9 reference with i5 reference subgrids:" << std::endl;
    get_accuracy(reference_gen9, reference_i5);

    std::cout << "Compare gen9 CPU reference with i5 reference subgrids:" << std::endl;
    get_accuracy(reference_gen9cpu, reference_i5);

    std::cout << "Compare iris reference with i5 reference subgrids:" << std::endl;
    get_accuracy(reference_iris, reference_i5);

    std::cout << "Compare iris CPU reference with i5 reference subgrids:" << std::endl;
    get_accuracy(reference_iriscpu, reference_i5);

    std::cout << "--------------------------------------------------------------------------------------\n" << std::endl;

    std::cout << "Compare i5 buffer reference with i5 reference subgrids:" << std::endl;
    get_accuracy(buffer_i5, reference_i5);

    std::cout << "Compare gold buffer reference with i5 reference subgrids:" << std::endl;
    get_accuracy(buffer_gold, reference_i5);

    std::cout << "Compare platinum buffer with i5 reference subgrids:" << std::endl;
    get_accuracy(buffer_platinum, reference_i5);

    std::cout << "Compare gen9 buffer with i5 reference subgrids:" << std::endl;
    get_accuracy(buffer_gen9, reference_i5);

    std::cout << "Compare gen9 CPU buffer with i5 reference subgrids:" << std::endl;
    get_accuracy(buffer_gen9cpu, reference_i5);

    std::cout << "Compare iris buffer with i5 reference subgrids:" << std::endl;
    get_accuracy(buffer_iris, reference_i5);

    std::cout << "Compare iris CPU buffer with i5 reference subgrids:" << std::endl;
    get_accuracy(buffer_iriscpu, reference_i5);

    std::cout << "--------------------------------------------------------------------------------------\n" << std::endl;

    std::cout << "Compare i5 implicit with i5 reference subgrids:" << std::endl;
    get_accuracy(implicit_i5, reference_i5);

    std::cout << "Compare gold implicit with i5 reference subgrids:" << std::endl;
    get_accuracy(implicit_gold, reference_i5);

    std::cout << "Compare platinum implicit with i5 reference subgrids:" << std::endl;
    get_accuracy(implicit_platinum, reference_i5);

    std::cout << "Compare gen9 implicit with i5 reference subgrids:" << std::endl;
    get_accuracy(implicit_gen9, reference_i5);

    std::cout << "Compare gen9 CPU implicit with i5 reference subgrids:" << std::endl;
    get_accuracy(implicit_gen9cpu, reference_i5);

    std::cout << "Compare iris implicit with i5 reference subgrids:" << std::endl;
    get_accuracy(implicit_iris, reference_i5);

    std::cout << "Compare iris CPU implicit with i5 reference subgrids:" << std::endl;
    get_accuracy(implicit_iriscpu, reference_i5);

    return EXIT_SUCCESS;
}
