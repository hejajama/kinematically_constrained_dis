// Tests of what the Sampling options of cross_sections.hpp rely on: the
// symmetries used to fold the integration ranges, the relative coordinates of
// the dipole pair, and the FE 5D integral integrated by parts in M_qq^2.

#include <cmath>
#include <functional>
#include <vector>

#include "cross_sections.hpp"
#include "dipole.hpp"
#include "doctest.h"
#include "integrands.hpp"
#include "interpolation.hpp"
#include "test_helpers.h"

namespace {

const dis::Kinematics kin{std::sqrt(2.0), 1e-3, 0.14, std::sqrt(2.0 / 3.0), 2 * 2.57 * 18.81, 0.01};

using Dipole = dis::BKDipole<dis::BicubicSpline>;

const Dipole &mv_dipole() {
    static const Dipole dipole(DIS_BKFILE);
    return dipole;
}

// Composite Gauss-Legendre rule: `panels` equal panels of `n` nodes on [a, b].
// Nodes from Newton's method on the Legendre polynomial.
double gauss_legendre(const std::function<double(double)> &f, double a, double b, int panels,
                      int n = 12) {
    std::vector<double> x(n), w(n);
    for (int i = 0; i < n; ++i) {
        double t = std::cos(dis::pi * (i + 0.75) / (n + 0.5)), dp = 0;
        for (int iter = 0; iter < 100; ++iter) {
            double p0 = 1, p1 = t;
            for (int j = 2; j <= n; ++j) {
                const double p2 = ((2 * j - 1) * t * p1 - (j - 1) * p0) / j;
                p0 = p1;
                p1 = p2;
            }
            dp = n * (t * p1 - p0) / (t * t - 1);
            const double step = p1 / dp;
            t -= step;
            if (std::abs(step) < 1e-16) break;
        }
        x[i] = t;
        w[i] = 2 / ((1 - t * t) * dp * dp);
    }
    double sum = 0;
    const double h = (b - a) / panels;
    for (int p = 0; p < panels; ++p) {
        const double mid = a + (p + 0.5) * h;
        for (int i = 0; i < n; ++i) sum += w[i] * h / 2 * f(mid + h / 2 * x[i]);
    }
    return sum;
}

}  // namespace

TEST_CASE("gauss_legendre helper integrates exactly") {
    CHECK(gauss_legendre([](double x) { return std::cos(x); }, 0, 2, 4) ==
          doctest::Approx(std::sin(2.0)).epsilon(1e-14));
}

TEST_CASE("FE integrands are symmetric under z -> 1 - z and theta -> 2 pi - theta") {
    const ReferenceTable ref("integrand_FE5D.txt");
    for (bool largeNc : {false, true}) {
        const dis::FE4DIntegrand<Dipole> f4(kin, mv_dipole(), largeNc);
        const dis::FE5DIntegrand<Dipole> f5(kin, mv_dipole(), largeNc);
        std::vector<double> a, b_z, b_theta;
        for (std::size_t i = 0; i < ref.size(); i += 4) {
            const double u = ref(i, "u"), up = ref(i, "up"), z = ref(i, "z"), th = ref(i, "theta");
            const double M2 = ref(i, "Mqq_sq");
            a.push_back(f4.evaluate(u, up, z, th).T);
            b_z.push_back(f4.evaluate(u, up, 1 - z, th).T);
            b_theta.push_back(f4.evaluate(u, up, z, 2 * dis::pi - th).T);
            a.push_back(f5.evaluate(u, up, z, th, M2).L);
            b_z.push_back(f5.evaluate(u, up, 1 - z, th, M2).L);
            b_theta.push_back(f5.evaluate(u, up, z, 2 * dis::pi - th, M2).L);
        }
        // 1 - z loses digits of z near 1, so not to round-off.
        check_columns_close("z -> 1-z", b_z, a, 1e-6, 1e-12 * max_abs(a));
        check_columns_close("theta -> -theta", b_theta, a, 1e-9, 1e-14 * max_abs(a));
    }
}

TEST_CASE("Relative coordinates describe the same dipole pair as polar ones") {
    const dis::FE4DIntegrand<Dipole> f(kin, mv_dipole(), false);
    for (double u : {0.01, 0.4, 3.0})
        for (double r : {1e-7, 0.05, 1.0, 5.0})
            for (double phi : {0.1, 1.5, 3.0}) {
                const auto rel = dis::DipolePair::relative(u, r, phi);
                const double upx = u - r * std::cos(phi), upy = -r * std::sin(phi);
                const double theta = std::atan2(upy, upx);
                CHECK(rel.up == doctest::Approx(std::hypot(upx, upy)).epsilon(1e-15));
                CHECK(rel.r2 == r * r);
                if (r < 1e-3) continue;  // polar r^2 = u^2 + u'^2 - ... cancels
                const auto polar = f.evaluate(u, rel.up, 0.3, theta);
                const auto relative = f.evaluate(rel, 0.3);
                CHECK(relative.L == doctest::Approx(polar.L).epsilon(1e-9));
                CHECK(relative.T == doctest::Approx(polar.T).epsilon(1e-9));
            }
}

TEST_CASE("Relative coordinates: points with u' outside the range are rejected") {
    const dis::FE4DIntegrand<Dipole> f(kin, mv_dipole(), false, dis::FECoordinates::relative,
                                       1e-6, 10.0);
    double out[2];
    const double inside[4] = {1.0, 0.5, 1.0, 0.3};    // u' ~ 0.88
    const double outside[4] = {9.0, 5.0, 3.0, 0.3};  // u' ~ 14
    f(inside, out);
    CHECK(out[0] != 0.0);
    f(outside, out);
    CHECK(out[0] == 0.0);
    CHECK(out[1] == 0.0);
}

TEST_CASE("FE 5D by parts: boundary term + correction = the M_qq^2 integral") {
    // At fixed u, u', theta, z both sides are 1D integrals over M_qq^2, done
    // here by quadrature in k = sqrt(z (1-z) M^2 - mf^2), dM^2 = 2k dk / (z(1-z)),
    // with panels split where the dipole's rapidity reaches Y_min (a kink).
    struct Point {
        double u, up, theta, z;
    };
    const Point points[] = {
        {0.5, 0.7, 0.3, 0.3}, {1.2, 1.1, 2.5, 0.1}, {2.0, 2.05, 0.02, 0.45}, {0.05, 3.0, 1.0, 0.2}};
    const double W2 = kin.W2(), m2 = kin.mf * kin.mf;

    for (bool largeNc : {false, true}) {
        const dis::FE5DIntegrand<Dipole> full(kin, mv_dipole(), largeNc);
        const dis::FE4DIntegrand<Dipole> boundary(kin, mv_dipole(), std::log(kin.x0), largeNc,
                                                  dis::FECoordinates::polar, 0, INFINITY);
        const dis::FE5DCorrectionIntegrand<Dipole> correction(kin, mv_dipole(), largeNc);

        for (const auto &p : points) {
            const double zz = p.z * (1 - p.z);
            auto k_of = [&](double M2) { return std::sqrt(std::max(zz * M2 - m2, 0.0)); };
            auto M2_of = [&](double k) { return (k * k + m2) / zz; };
            const double K = k_of(W2);
            const double k_kink = k_of(correction.Mqq_sq_hi());  // Y(M^2) = Y_min

            for (int pol = 0; pol < 2; ++pol) {
                auto component = [&](const dis::LT &v) { return pol == 0 ? v.L : v.T; };
                auto f_full = [&](double k) {
                    return component(full.evaluate(p.u, p.up, p.z, p.theta, M2_of(k))) * 2 * k / zz;
                };
                auto f_corr = [&](double k) {
                    return component(correction.evaluate(p.u, p.up, p.z, p.theta, M2_of(k))) * 2 *
                           k / zz;
                };
                const double original =
                    gauss_legendre(f_full, 0, k_kink, 400) + gauss_legendre(f_full, k_kink, K, 2000);
                const double by_parts = component(boundary.evaluate(p.u, p.up, p.z, p.theta)) +
                                        gauss_legendre(f_corr, 0, k_kink, 400);
                INFO("u " << p.u << " up " << p.up << " theta " << p.theta << " z " << p.z
                          << (pol == 0 ? " L" : " T") << (largeNc ? " largeNc" : ""));
                CHECK(by_parts == doctest::Approx(original).epsilon(1e-7));
                // The correction vanishes above the kink: the dipole is frozen there.
                CHECK(f_corr(0.5 * (k_kink + K)) == 0.0);
            }
        }
    }
}

TEST_CASE("Sampling::python() turns every option off") {
    const auto s = dis::Sampling::python();
    CHECK(!s.log_sizes);
    CHECK(!s.relative_coordinates);
    CHECK(!s.log_relative);
    CHECK(!s.log_Mqq);
    CHECK(!s.symmetries);
    CHECK(!s.fe5d_by_parts);
}
