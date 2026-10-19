#include <cmath>
#include <random>
#include <vector>

#include "doctest.h"
#include "interpolation.hpp"
#include "rcbk_table.hpp"

namespace {

// A table on a nonuniform grid (neither uniform nor geometric, so the
// locator falls back to binary search) filled from f(r, Y).
template <class F>
dis::RcbkTable synthetic_table(F f) {
    dis::RcbkTable t;
    t.filename = "synthetic";
    t.r_grid = {0.1, 0.25, 0.3, 0.7, 1.1, 1.2, 2.0, 3.5};
    t.Y_grid = {0.0, 0.5, 0.6, 1.4, 2.0, 2.2};
    for (double Y : t.Y_grid)
        for (double r : t.r_grid) t.N_grid.push_back(f(r, Y));
    return t;
}

double bicubic(double r, double Y) {
    return 0.3 + 0.2 * r - 0.1 * r * r + 0.05 * r * r * r + Y * (0.4 - 0.3 * r * r) +
           Y * Y * (0.1 + 0.02 * r * r * r) - 0.03 * Y * Y * Y * r;
}

double bilinear(double r, double Y) { return 0.3 + 0.2 * r - 0.4 * Y + 0.7 * r * Y; }

}  // namespace

TEST_CASE("not_a_knot_slopes: exact for cubic data on a nonuniform grid") {
    const std::vector<double> x = {-1.0, -0.3, 0.2, 0.25, 1.0, 2.5, 2.6};
    std::vector<double> y;
    for (double xi : x) y.push_back(1 - 2 * xi + 0.5 * xi * xi - 0.7 * xi * xi * xi);
    const auto m = dis::not_a_knot_slopes(x, y);
    for (std::size_t i = 0; i < x.size(); ++i)
        CHECK(m[i] == doctest::Approx(-2 + x[i] - 2.1 * x[i] * x[i]).epsilon(1e-12));

    CHECK_THROWS(dis::not_a_knot_slopes({0, 1, 2}, {0, 1, 2}));
}

TEST_CASE("CellLocator: uniform, geometric and general grids") {
    const dis::CellLocator uniform({0.0, 0.2, 0.4, 0.6, 0.8});
    CHECK(uniform(0.0) == 0);
    CHECK(uniform(0.3) == 1);
    CHECK(uniform(0.79) == 3);
    CHECK(uniform(0.8) == 3);   // last node: clipped into the last cell
    CHECK(uniform(-1.0) == 0);  // below: clipped
    CHECK(uniform(5.0) == 3);

    const dis::CellLocator geometric({1e-3, 1e-2, 1e-1, 1.0});
    CHECK(geometric(5e-3) == 0);
    CHECK(geometric(0.05) == 1);
    CHECK(geometric(0.5) == 2);
    CHECK(geometric(1.0) == 2);

    const dis::CellLocator general({0.0, 0.1, 1.0, 1.5});
    CHECK(general(0.05) == 0);
    CHECK(general(0.1) == 1);
    CHECK(general(1.2) == 2);
    CHECK(general(3.0) == 2);
}

TEST_CASE("BicubicSpline: reproduces a bicubic polynomial exactly") {
    const auto table = synthetic_table(bicubic);
    const dis::BicubicSpline spline(table);
    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> r_dist(0.1, 3.5), Y_dist(0.0, 2.2);
    for (int i = 0; i < 200; ++i) {
        const double r = r_dist(rng), Y = Y_dist(rng);
        CHECK(spline.at(Y)(r) == doctest::Approx(bicubic(r, Y)).epsilon(1e-12));
    }
}

TEST_CASE("BicubicSpline: passes through every node of mv.dat") {
    const auto table = dis::RcbkTable::read(DIS_BKFILE);
    const dis::BicubicSpline spline(table);
    double worst = 0;
    for (std::size_t iY = 0; iY < table.n_Y(); ++iY) {
        const auto slice = spline.at(table.Y_grid[iY]);
        for (std::size_t ir = 0; ir < table.n_r(); ++ir)
            worst = std::max(worst, std::abs(slice(table.r_grid[ir]) - table.N(iY, ir)));
    }
    CHECK(worst < 1e-13);
}

TEST_CASE("BicubicSpline: rejects grids that are too short") {
    dis::RcbkTable t;
    t.r_grid = {1, 2, 3};
    t.Y_grid = {0, 1, 2, 3};
    t.N_grid.assign(12, 0.5);
    CHECK_THROWS(dis::BicubicSpline(t));
}

TEST_CASE("Bilinear: reproduces a bilinear function exactly and passes through nodes") {
    const auto table = synthetic_table(bilinear);
    const dis::Bilinear interp(table);
    std::mt19937_64 rng(2);
    std::uniform_real_distribution<double> r_dist(0.1, 3.5), Y_dist(0.0, 2.2);
    for (int i = 0; i < 200; ++i) {
        const double r = r_dist(rng), Y = Y_dist(rng);
        CHECK(interp.at(Y)(r) == doctest::Approx(bilinear(r, Y)).epsilon(1e-13));
    }

    const auto mv = dis::RcbkTable::read(DIS_BKFILE);
    const dis::Bilinear mv_interp(mv);
    for (std::size_t iY = 0; iY < mv.n_Y(); iY += 7)
        for (std::size_t ir = 0; ir < mv.n_r(); ir += 13)
            CHECK(mv_interp.at(mv.Y_grid[iY])(mv.r_grid[ir]) ==
                  doctest::Approx(mv.N(iY, ir)).epsilon(1e-13));
}
