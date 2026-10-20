#include "filon.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "special_functions.hpp"

namespace dis {

namespace {

// Below this x = r sqrt(s) the antiderivative is summed as a power series;
// above it the recurrence for int x^m J0 is used. The recurrence cancels
// badly at small x (its terms are ~x^(m-1) while the result is ~x^(m+1)),
// the series converges slowly at large x; at x = 4 both are accurate.
constexpr double series_below = 4.0;

// int_0^s t^n J0(r sqrt(t)) dt for n = 0..3 by the power series
// s^(n+1) sum_j (-1)^j (x^2/4)^j / ((j!)^2 (n + j + 1)), x = r sqrt(s).
std::array<double, 4> antiderivatives_series(double s, double x) {
    const double q = -x * x / 4;
    std::array<double, 4> sum = {1.0, 1.0 / 2, 1.0 / 3, 1.0 / 4};
    double term = 1;
    for (int j = 1; j < 60; ++j) {
        term *= q / (double(j) * j);
        double largest = 0;
        for (int n = 0; n < 4; ++n) {
            const double add = term / (n + j + 1);
            sum[n] += add;
            largest = std::max(largest, std::abs(add / sum[n]));
        }
        if (largest < 1e-17) break;
    }
    double s_pow = s;
    for (int n = 0; n < 4; ++n, s_pow *= s) sum[n] *= s_pow;
    return sum;
}

// The same from int_0^x y^m J0(y) dy for m = 1, 3, 5, 7:
//   I_m = x^m J1 + (m-1) x^(m-1) J0 - (m-1)^2 I_(m-2),  I_1 = x J1,
// and t = y^2 / r^2:  int_0^s t^n J0(r sqrt t) dt = 2 I_(2n+1)(x) / r^(2n+2).
std::array<double, 4> antiderivatives_recurrence(double x, double r) {
    const double J0 = bessel_J0(x), J1 = bessel_J1(x);
    std::array<double, 4> A{};
    double I = x * J1, x_pow = x;  // x^m
    const double inv_r2 = 1 / (r * r);
    double scale = 2 * inv_r2;  // 2 / r^(2n+2)
    A[0] = scale * I;
    for (int n = 1; n < 4; ++n) {
        const int m = 2 * n + 1;
        const double xm1 = x_pow * x;  // x^(m-1)
        x_pow = xm1 * x;               // x^m
        I = x_pow * J1 + (m - 1) * xm1 * J0 - (m - 1) * (m - 1) * I;
        scale *= inv_r2;
        A[n] = scale * I;
    }
    return A;
}

}  // namespace

std::array<double, 4> j0_moment_antiderivatives(double s, double r) {
    if (s <= 0) return {0, 0, 0, 0};
    const double x = r * std::sqrt(s);
    return x < series_below ? antiderivatives_series(s, x) : antiderivatives_recurrence(x, r);
}

double j0_moment_antiderivative(int n, double s, double r) {
    if (n < 0 || n > 3) throw std::invalid_argument("j0_moment_antiderivative: n must be 0..3");
    return j0_moment_antiderivatives(s, r)[n];
}

std::array<double, 4> filon_weights(double s_a, double s_b, double r) {
    // Lagrange basis on the nodes t = -1, -1/2, 1/2, 1: L_j(t) = sum_n a[j][n] t^n.
    static constexpr double a[4][4] = {
        {-1.0 / 6, 1.0 / 6, 2.0 / 3, -2.0 / 3},
        {2.0 / 3, -4.0 / 3, -2.0 / 3, 4.0 / 3},
        {2.0 / 3, 4.0 / 3, -2.0 / 3, -4.0 / 3},
        {-1.0 / 6, -1.0 / 6, 2.0 / 3, 2.0 / 3},
    };
    static constexpr double binomial[4][4] = {{1, 0, 0, 0}, {1, 1, 0, 0}, {1, 2, 1, 0}, {1, 3, 3, 1}};

    const double s_c = 0.5 * (s_a + s_b), h = 0.5 * (s_b - s_a);
    // Moments in s, then in t = (s - s_c) / h:
    //   nu_n = int t^n J0 ds = h^-n sum_k C(n,k) (-s_c)^(n-k) mu_k.
    const auto A_b = j0_moment_antiderivatives(s_b, r), A_a = j0_moment_antiderivatives(s_a, r);
    double mu[4], nu[4];
    for (int k = 0; k < 4; ++k) mu[k] = A_b[k] - A_a[k];
    for (int n = 0; n < 4; ++n) {
        double sum = 0;
        for (int k = 0; k <= n; ++k) sum += binomial[n][k] * std::pow(-s_c, n - k) * mu[k];
        nu[n] = sum / std::pow(h, n);
    }
    std::array<double, 4> w{};
    for (int j = 0; j < 4; ++j)
        for (int n = 0; n < 4; ++n) w[j] += a[j][n] * nu[n];
    return w;
}

}  // namespace dis
