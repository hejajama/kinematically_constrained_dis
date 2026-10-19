#include <cmath>
#include <string>
#include <vector>

#include "dipole.hpp"
#include "doctest.h"
#include "integrands.hpp"
#include "interpolation.hpp"
#include "test_helpers.h"

namespace {

// Parameters of the reference files (defaults of test_bk_integration.py, xB = 1e-3).
const dis::Kinematics kin{std::sqrt(2.0), 1e-3, 0.14, std::sqrt(2.0 / 3.0), 2 * 2.57 * 18.81, 0.01};

using Dipole = dis::BKDipole<dis::BicubicSpline>;

const Dipole &mv_dipole() {
    static const Dipole dipole(DIS_BKFILE);
    return dipole;
}

// Integrands are compared relative to their own scale: they span many orders
// of magnitude and change sign (FE), so a pure relative tolerance would
// fail near their zeros, where the inputs to J0, J1 and cos(theta) amplify
// the round-off differences between numpy and C++.
void check_integrand_column(const std::string &what, const std::vector<double> &got,
                            const std::vector<double> &want) {
    check_columns_close(what, got, want, 1e-9, 1e-13 * max_abs(want));
}

}  // namespace

TEST_CASE("OT integrand matches Python (integrand_OT.txt)") {
    const dis::OTIntegrand<Dipole> f(kin, mv_dipole());
    const ReferenceTable ref("integrand_OT.txt");
    std::vector<double> L, T;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const auto v = f.evaluate(ref(i, "r"), ref(i, "z"));
        L.push_back(v.L);
        T.push_back(v.T);
    }
    check_integrand_column("OT L", L, ref.column("f_L"));
    check_integrand_column("OT T", T, ref.column("f_T"));
}

TEST_CASE("FE 4D integrand matches Python (integrand_FE4D.txt)") {
    const ReferenceTable ref("integrand_FE4D.txt");
    for (bool largeNc : {false, true}) {
        const dis::FE4DIntegrand<Dipole> f(kin, mv_dipole(), largeNc);
        const std::string nc = largeNc ? "largeNc" : "finiteNc";
        std::vector<double> L, T;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            const auto v = f.evaluate(ref(i, "u"), ref(i, "up"), ref(i, "z"), ref(i, "theta"));
            L.push_back(v.L);
            T.push_back(v.T);
        }
        check_integrand_column("FE4D L " + nc, L, ref.column("f_L_" + nc));
        check_integrand_column("FE4D T " + nc, T, ref.column("f_T_" + nc));
    }
}

TEST_CASE("FE 5D integrand matches Python (integrand_FE5D.txt)") {
    const ReferenceTable ref("integrand_FE5D.txt");
    for (bool largeNc : {false, true}) {
        const dis::FE5DIntegrand<Dipole> f(kin, mv_dipole(), largeNc);
        const std::string nc = largeNc ? "largeNc" : "finiteNc";
        std::vector<double> L, T;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            const auto v = f.evaluate(ref(i, "u"), ref(i, "up"), ref(i, "z"), ref(i, "theta"),
                                      ref(i, "Mqq_sq"));
            L.push_back(v.L);
            T.push_back(v.T);
        }
        check_integrand_column("FE5D L " + nc, L, ref.column("f_L_" + nc));
        check_integrand_column("FE5D T " + nc, T, ref.column("f_T_" + nc));
    }
}

TEST_CASE("Reference rows cover the special regions of the FE 5D integrand") {
    // Guards against a regenerated reference file losing its coverage.
    const ReferenceTable ref("integrand_FE5D.txt");
    const double W2 = kin.W2(), Q2 = kin.Q * kin.Q;
    int forbidden = 0, clamped_Y = 0, evolved = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double z = ref(i, "z"), M2 = ref(i, "Mqq_sq");
        if (M2 < kin.mf * kin.mf / (z * (1 - z))) {
            ++forbidden;
            CHECK(ref(i, "f_L_finiteNc") == 0.0);
        } else if ((M2 + Q2) / (W2 + Q2) > kin.x0) {
            ++clamped_Y;
        } else {
            ++evolved;
        }
    }
    CHECK(forbidden > 50);
    CHECK(clamped_Y > 50);
    CHECK(evolved > 50);
}

TEST_CASE("OT integrand is symmetric under z -> 1 - z") {
    const dis::OTIntegrand<Dipole> f(kin, mv_dipole());
    for (double r : {1e-3, 0.3, 2.0})
        for (double z : {1e-5, 0.1, 0.37}) {
            CHECK(f.evaluate(r, z).L == doctest::Approx(f.evaluate(r, 1 - z).L).epsilon(1e-9));
            CHECK(f.evaluate(r, z).T == doctest::Approx(f.evaluate(r, 1 - z).T).epsilon(1e-9));
        }
}

TEST_CASE("Integrand operator() returns L and T of evaluate()") {
    const dis::FE5DIntegrand<Dipole> f(kin, mv_dipole(), false);
    const double x[5] = {0.7, 1.3, 2.0, 0.4, 50.0};  // (u, up, theta, z, Mqq_sq)
    double out[2];
    f(x, out);
    const auto v = f.evaluate(x[0], x[1], x[3], x[2], x[4]);
    CHECK(out[0] == v.L);
    CHECK(out[1] == v.T);
    CHECK(v.L != 0.0);
}
