// The Filon rule for int ds J0(r sqrt(s)) T(s) and the FE 5D integrand that
// uses it, against direct quadrature.

#include <cmath>
#include <string>
#include <vector>

#include "dipole.hpp"
#include "doctest.h"
#include "cross_sections.hpp"
#include "filon.hpp"
#include "integrands.hpp"
#include "interpolation.hpp"
#include "special_functions.hpp"
#include "test_helpers.h"

namespace {

const dis::Kinematics kin{std::sqrt(2.0), 1e-3, 0.14, std::sqrt(2.0 / 3.0), 2 * 2.57 * 18.81, 0.01};

using BK = dis::BKDipole<dis::BicubicSpline>;
using Extrapolated = dis::LargeXExtrapolation<BK>;

const BK &mv_dipole() {
    static const BK dipole(DIS_BKFILE);
    return dipole;
}

// int_0^s t^n J0(r sqrt t) dt by quadrature in k = sqrt(t), dt = 2k dk.
double moment_by_quadrature(int n, double s, double r) {
    const int panels = 20 + static_cast<int>(4 * r * std::sqrt(s));
    return gauss_legendre(
        [&](double k) { return 2 * std::pow(k, 2 * n + 1) * dis::bessel_J0(r * k); }, 0,
        std::sqrt(s), panels);
}

}  // namespace

TEST_CASE("j0_moment_antiderivative matches quadrature, in both of its regimes") {
    for (int n = 0; n <= 3; ++n)
        for (double r : {1e-6, 0.01, 0.3, 2.0, 10.0})
            for (double s : {0.5, 10.0, 300.0}) {
                INFO("n " << n << " r " << r << " s " << s << " x " << r * std::sqrt(s));
                const double exact = moment_by_quadrature(n, s, r);
                CHECK(dis::j0_moment_antiderivative(n, s, r) ==
                      doctest::Approx(exact).epsilon(1e-10).scale(1e-12 * std::pow(s, n + 1)));
            }
    CHECK(dis::j0_moment_antiderivative(2, 0.0, 1.0) == 0.0);
}

TEST_CASE("j0_moment_antiderivative is continuous where it switches from series to recurrence") {
    for (int n = 0; n <= 3; ++n)
        for (double r : {0.5, 3.0}) {
            const double s = std::pow(4.0 / r, 2);  // x = r sqrt(s) = 4
            const double below = dis::j0_moment_antiderivative(n, s * (1 - 1e-13), r);
            const double above = dis::j0_moment_antiderivative(n, s * (1 + 1e-13), r);
            CHECK(above == doctest::Approx(below).epsilon(1e-11));
        }
}

TEST_CASE("filon_weights integrate J0 times a cubic exactly") {
    auto p = [](double s) { return 1 + 0.3 * s - 0.02 * s * s + 0.0011 * s * s * s; };
    for (double r : {0.01, 0.7, 3.0, 20.0})
        for (const auto &range : {std::pair{0.0, 2.0}, {2.0, 50.0}, {400.0, 460.0}}) {
            const double s_a = range.first, s_b = range.second;
            const auto w = dis::filon_weights(s_a, s_b, r);
            const double s_c = 0.5 * (s_a + s_b), h = 0.5 * (s_b - s_a);
            double filon = 0;
            for (int j = 0; j < 4; ++j) filon += w[j] * p(s_c + h * dis::filon_nodes[j]);
            const int panels = 20 + static_cast<int>(4 * r * std::sqrt(s_b));
            const double exact = gauss_legendre(
                [&](double k) { return 2 * k * dis::bessel_J0(r * k) * p(k * k); }, std::sqrt(s_a),
                std::sqrt(s_b), panels);
            INFO("r " << r << " [" << s_a << ", " << s_b << "]");
            CHECK(filon == doctest::Approx(exact).epsilon(1e-9).scale(1e-9 * std::abs(p(s_b)) * (s_b - s_a)));
        }
}

TEST_CASE("LargeXExtrapolation: N(r, x) = N(r, x0) [(1 - x) / (1 - x0)]^power for x > x0") {
    const Extrapolated ext(mv_dipole(), kin.x0, 4.0);
    for (double r : {0.01, 0.5, 3.0}) {
        // Y >= 0: the BK dipole itself.
        CHECK(ext.at(1.3).S(r) == mv_dipole().at(1.3).S(r));
        CHECK(ext.at(0.0).S(r) == mv_dipole().at(0.0).S(r));
        // x = 0.3 > x0: Y = ln(x0 / 0.3) < 0.
        const double Y = std::log(kin.x0 / 0.3);
        CHECK(ext.at(Y).N(r) ==
              doctest::Approx(mv_dipole().at(0.0).N(r) * std::pow(0.7 / (1 - kin.x0), 4)));
        // Continuous at x = x0.
        CHECK(ext.at(-1e-12).N(r) == doctest::Approx(mv_dipole().at(0.0).N(r)).epsilon(1e-10));
        // x = 1: transparent.
        CHECK(ext.at(std::log(kin.x0)).S(r) == 1.0);
    }
    CHECK(ext.Y_min() == doctest::Approx(std::log(kin.x0)));
    CHECK(ext.Y_max() == mv_dipole().Y_max());
    const auto b = ext.Y_breakpoints();
    REQUIRE(b.size() == 3);
    CHECK(b[1] == 0.0);
}

TEST_CASE("FE 5D by Filon = the M_qq^2 integral by direct quadrature") {
    // At fixed u, u', theta, z: the Filon integrand against the M_qq^2 integral
    // of FE5DIntegrand done by brute force in k = sqrt(z(1-z) M^2 - mf^2), with
    // panels split where the dipole has a kink (Y = 0, i.e. xP = x0).
    struct Point {
        double u, up, theta, z;
    };
    const Point points[] = {{0.5, 0.7, 0.3, 0.3},
                            {1.2, 1.1, 2.5, 0.1},
                            {2.0, 2.05, 0.02, 0.45},
                            {0.05, 3.0, 1.0, 0.2},
                            {4.0, 6.0, 2.0, 0.05}};
    const double W2 = kin.W2(), m2 = kin.mf * kin.mf, Q2 = kin.Q * kin.Q;
    const Extrapolated extrapolated(mv_dipole(), kin.x0, 4.0);

    auto check = [&](const auto &dipole, const char *name) {
        using D = std::decay_t<decltype(dipole)>;
        for (bool largeNc : {false, true}) {
            const dis::FE5DIntegrand<D> full(kin, dipole, largeNc);
            // Higher resolution than the default, to test convergence to the
            // right value; the default 8 nodes and 8 panels are checked below.
            const dis::FE5DFilonIntegrand<D> filon(kin, dipole, largeNc, dis::FECoordinates::polar,
                                                   0, INFINITY, 16, 32);
            const dis::FE5DFilonIntegrand<D> filon_default(kin, dipole, largeNc);
            for (const auto &p : points) {
                const double zz = p.z * (1 - p.z);
                auto M2_of = [&](double k) { return (k * k + m2) / zz; };
                const double K = std::sqrt(zz * W2 - m2);
                const double k_kink = std::sqrt(zz * (kin.x0 * (W2 + Q2) - Q2) - m2);
                const auto f = filon.evaluate(p.u, p.up, p.z, p.theta);
                const auto f_default = filon_default.evaluate(p.u, p.up, p.z, p.theta);
                for (int pol = 0; pol < 2; ++pol) {
                    auto integrand = [&](double k) {
                        const auto v = full.evaluate(p.u, p.up, p.z, p.theta, M2_of(k));
                        return (pol == 0 ? v.L : v.T) * 2 * k / zz;
                    };
                    const double below = gauss_legendre(integrand, 0, k_kink, 200);
                    const double above = gauss_legendre(integrand, k_kink, K, 2000);
                    // Relative to int |integrand|, the size of the contributions:
                    // with many oscillations the integral itself is a near-total
                    // cancellation of them.
                    auto magnitude = [&](double k) { return std::abs(integrand(k)); };
                    const double scale = gauss_legendre(magnitude, 0, k_kink, 200) +
                                         gauss_legendre(magnitude, k_kink, K, 2000);
                    INFO(std::string(name) << " u " << p.u << " up " << p.up << " theta " << p.theta
                                           << " z " << p.z << (pol == 0 ? " L" : " T")
                                           << (largeNc ? " largeNc" : ""));
                    CHECK(std::abs((pol == 0 ? f.L : f.T) - (below + above)) <= 1e-5 * scale);
                    CHECK(std::abs((pol == 0 ? f_default.L : f_default.T) - (below + above)) <=
                          1e-3 * scale);
                }
            }
        }
    };
    check(mv_dipole(), "frozen");
    check(extrapolated, "extrapolated");
}

TEST_CASE("transparent: the extrapolated dipole at xP = 1, not the BK one") {
    const Extrapolated ext(mv_dipole(), kin.x0, 4.0);
    const dis::IntegrationLimits lim;
    CHECK(dis::transparent(ext.at(std::log(kin.x0)), lim));
    CHECK(!dis::transparent(ext.at(std::log(kin.x0) + 0.1), lim));
    CHECK(!dis::transparent(mv_dipole().at(std::log(kin.x0)), lim));
}
