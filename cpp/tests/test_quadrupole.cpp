#include <cmath>
#include <string>
#include <vector>

#include "dipole.hpp"
#include "doctest.h"
#include "interpolation.hpp"
#include "quadrupole.hpp"
#include "test_helpers.h"

namespace {

// Evaluates S(u), S(u'), S4 (finite and large Nc) on every row of a
// reference file and compares with its columns. `dipole_at` gives the dipole
// at the row's rapidity.
template <class DipoleAt>
void compare_with_reference(const ReferenceTable &ref, DipoleAt dipole_at) {
    std::vector<double> Su, Sup, S4F, S4L;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const auto S = dipole_at(i);
        const double u = ref(i, "u"), up = ref(i, "up"), z = ref(i, "z"), th = ref(i, "theta");
        const auto finite = dis::polar_correlators(S, u, up, z, th, false);
        const auto large = dis::polar_correlators(S, u, up, z, th, true);
        Su.push_back(finite.S_u);
        Sup.push_back(finite.S_up);
        S4F.push_back(finite.S4);
        S4L.push_back(large.S4);
    }
    // The correlators are O(1) combinations of exp and log of the dipoles;
    // absolute tolerances cover the rows where S4 crosses zero.
    check_columns_close("S_u", Su, ref.column("S_u"), 1e-12, 1e-14);
    check_columns_close("S_up", Sup, ref.column("S_up"), 1e-12, 1e-14);
    check_columns_close("S4 finite Nc", S4F, ref.column("S4_finiteNc"), 1e-10, 1e-12);
    check_columns_close("S4 large Nc", S4L, ref.column("S4_largeNc"), 1e-10, 1e-12);
}

}  // namespace

TEST_CASE("Quadrupole with a GBW dipole matches Python (quadrupole_gbw.txt)") {
    const dis::GBWDipole gbw(1.0);
    compare_with_reference(ReferenceTable("quadrupole_gbw.txt"),
                           [&](std::size_t) { return gbw.at(0.0); });
}

TEST_CASE("Quadrupole with the BK dipole at per-point Y matches Python (quadrupole_bk.txt)") {
    const dis::BKDipole<dis::BicubicSpline> bk(DIS_BKFILE);
    const ReferenceTable ref("quadrupole_bk.txt");
    compare_with_reference(ref, [&](std::size_t i) { return bk.at(ref(i, "Y")); });
}

TEST_CASE("Quadrupole: transparent target gives S4 = 1") {
    // All six dipoles equal to 1: the Delta = 0 branch.
    const dis::GBWDipole transparent(0.0);
    for (bool largeNc : {false, true}) {
        const auto c = dis::polar_correlators(transparent.at(0), 1.3, 0.4, 0.3, 1.0, largeNc);
        CHECK(c.S_u == 1.0);
        CHECK(c.S_up == 1.0);
        CHECK(c.S4 == doctest::Approx(1.0).epsilon(1e-12));
    }
}

TEST_CASE("Quadrupole: x1 = x1', x2 = x2' gives S4 = 1") {
    // u = u', theta = 0: Tr(V1 V2^dag V2 V1^dag) / Nc = 1 for any target.
    const dis::BKDipole<dis::BicubicSpline> bk(DIS_BKFILE);
    const dis::GBWDipole gbw(1.0);
    for (double u : {0.01, 0.5, 2.0, 6.0})
        for (double z : {0.1, 0.5}) {
            CHECK(dis::polar_correlators(bk.at(3.0), u, u, z, 0.0, false).S4 ==
                  doctest::Approx(1.0).epsilon(1e-9));
            CHECK(dis::polar_correlators(gbw.at(0), u, u, z, 0.0, false).S4 ==
                  doctest::Approx(1.0).epsilon(1e-9));
        }
}

TEST_CASE("pair_dipoles_polar: the six distances") {
    // A dipole that returns its argument, to see the distances themselves.
    struct Identity {
        double S(double r) const { return r; }
    };
    const double u = 2.0, up = 1.0, z = 0.25, theta = dis::pi / 2;
    const auto d = dis::pair_dipoles_polar(Identity{}, u, up, z, theta);
    // x1 = (1.5, 0), x2 = (-0.5, 0), x1' = (0, 0.75), x2' = (0, -0.25)
    CHECK(d.d12 == doctest::Approx(2.0));
    CHECK(d.d1p2p == doctest::Approx(1.0));
    CHECK(d.d11p == doctest::Approx(std::hypot(1.5, 0.75)));
    CHECK(d.d22p == doctest::Approx(std::hypot(0.5, 0.25)));
    CHECK(d.d12p == doctest::Approx(std::hypot(1.5, 0.25)));
    CHECK(d.d21p == doctest::Approx(std::hypot(0.5, 0.75)));
}
