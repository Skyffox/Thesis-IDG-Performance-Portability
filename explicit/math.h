#include <cmath>
#include <array>
#include <complex>

#define NR_POLARIZATIONS 4


inline float compute_l(
    int x,
    int subgrid_size,
    float image_size)
{
    return (x + 0.5 - (subgrid_size / 2)) * image_size / subgrid_size;
}

inline float compute_m(
    int y,
    int subgrid_size,
    float image_size)
{
    return compute_l(y, subgrid_size, image_size);
}

inline float compute_n(
    float l,
    float m)
{
    // evaluate n = 1.0f - sqrt(1.0 - (l * l) - (m * m));
    // accurately for small values of l and m
    const float tmp = (l * l) + (m * m);
    return tmp > 1.0 ? 1.0 : tmp / (1.0f + sqrtf(1.0f - tmp));
}

void matmul(
    const std::array<std::complex<float>, 4> a,
    const std::array<std::complex<float>, 4> b,
          std::array<std::complex<float>, 4> &c)
{
    c[0]  = a[0] * b[0];
    c[1]  = a[0] * b[1];
    c[2]  = a[2] * b[0];
    c[3]  = a[2] * b[1];
    c[0] += a[1] * b[2];
    c[1] += a[1] * b[3];
    c[2] += a[3] * b[2];
    c[3] += a[3] * b[3];
}

void conjugate(
    const std::array<std::complex<float>, 4> a,
          std::array<std::complex<float>, 4> &b)
{
    for (unsigned i = 0; i < 4; i++) {
        b[i] = std::conj(a[i]);
    }
}

void transpose(
    const std::array<std::complex<float>, 4> a,
          std::array<std::complex<float>, 4> &b)
{
    b[0] = a[0];
    b[1] = a[2];
    b[2] = a[1];
    b[3] = a[3];
}

void hermitian(
    const std::array<std::complex<float>, 4> a,
          std::array<std::complex<float>, 4> &b)
{
    std::array<std::complex<float>, 4> temp;
    conjugate(a, temp);
    transpose(temp, b);
}

void apply_aterm_gridder(
           std::array<std::complex<float>, 4> &pixels,
     const std::array<std::complex<float>, 4> aterm1,
     const std::array<std::complex<float>, 4> aterm2)
{
    // Aterm 1 hermitian
    std::array<std::complex<float>, 4> aterm1_h;
    hermitian(aterm1, aterm1_h);

    // Apply aterm: P = A1^H * P
    std::array<std::complex<float>, 4> temp;
    matmul(aterm1_h, pixels, temp);

    // Apply aterm: P = P * A2
    matmul(temp, aterm2, pixels);
}
