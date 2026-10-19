#include <cmath>
#include <vector>

#include "dipole.hpp"
#include "doctest.h"
#include "interpolation.hpp"
#include "test_helpers.h"

TEST_CASE("BKDipole<BicubicSpline>: N and S match Python BKDipole (dipole_N.txt)") {
    // Includes grid nodes, cell midpoints and queries outside the grid.
    const dis::BKDipole<dis::BicubicSpline> dipole(DIS_BKFILE);
    const ReferenceTable ref("dipole_N.txt");
    REQUIRE(ref.size() > 2000);

    std::vector<double> N, S;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        N.push_back(dipole.N(ref(i, "r"), ref(i, "Y")));
        S.push_back(dipole.S(ref(i, "r"), ref(i, "Y")));
    }
    // The spline is rebuilt here from the data, not copied from scipy, so it
    // agrees to round-off in N (and N ranges over [0, 1]).
    check_columns_close("N vs table", N, ref.column("N_table"), 0, 1e-13);
    check_columns_close("N vs FITPACK", N, ref.column("N_fitpack"), 0, 1e-13);
    check_columns_close("S", S, ref.column("S"), 0, 1e-13);
    // Where N is tiny (small r) also relative agreement.
    check_columns_close("N relative", N, ref.column("N_table"), 1e-8, 1e-20);
}

TEST_CASE("BKDipole: queries outside the grid are evaluated at the boundary") {
    const dis::BKDipole<dis::BicubicSpline> dipole(DIS_BKFILE);
    const auto &t = dipole.table();
    CHECK(dipole.N(1e-9, 3.0) == dipole.N(t.r_min(), 3.0));
    CHECK(dipole.N(1e4, 3.0) == dipole.N(t.r_max(), 3.0));
    CHECK(dipole.N(0.5, -2.0) == dipole.N(0.5, t.Y_min()));
    CHECK(dipole.N(0.5, 30.0) == dipole.N(0.5, t.Y_max()));
    CHECK(dipole.N(1e-9, -1.0) == dipole.N(t.r_min(), t.Y_min()));
}

TEST_CASE("BKDipole: N is clipped to [0, 1] and S is floored") {
    // A table whose data leave [0, 1], so the spline overshoots.
    dis::RcbkTable t;
    t.filename = "synthetic";
    t.r_grid = {0.1, 0.2, 0.3, 0.4, 0.5};
    t.Y_grid = {0.0, 1.0, 2.0, 3.0};
    for (int iY = 0; iY < 4; ++iY)
        for (double v : {-0.5, 0.2, 1.5, 0.9, 1.2}) t.N_grid.push_back(v);

    const dis::BKDipole<dis::BicubicSpline> clipped(t);
    const dis::BKDipole<dis::BicubicSpline> raw(t, dis::DipoleOptions{false, 1e-14});
    CHECK(raw.N(0.1, 1.5) == doctest::Approx(-0.5));
    CHECK(clipped.N(0.1, 1.5) == 0.0);
    CHECK(clipped.N(0.3, 1.5) == 1.0);
    CHECK(clipped.S(0.3, 1.5) == 1e-14);
    CHECK(raw.S(0.3, 1.5) == 1e-14);  // 1 - 1.5 < 0, floored
}

TEST_CASE("BKDipole<Bilinear>: close to the spline on mv.dat") {
    const dis::BKDipole<dis::BicubicSpline> spline(DIS_BKFILE);
    const dis::BKDipole<dis::Bilinear> linear(DIS_BKFILE);
    const ReferenceTable ref("dipole_N.txt");
    double worst = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double r = ref(i, "r"), Y = ref(i, "Y");
        worst = std::max(worst, std::abs(linear.N(r, Y) - spline.N(r, Y)));
    }
    // A different interpolation, so not equal; on this fine grid within 1e-3.
    CHECK(worst > 0);
    CHECK(worst < 1e-3);
}

TEST_CASE("GBWDipole: S(r) = exp(-r^2 Qs^2 / 4)") {
    const dis::GBWDipole gbw(2.0);
    CHECK(gbw.at(5.0).S(1.5) == doctest::Approx(std::exp(-1.5 * 1.5 * 2.0 / 4)));
    CHECK(gbw.at(0.0).S(0.0) == 1.0);
}
