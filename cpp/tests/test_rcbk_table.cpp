#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

#include "doctest.h"
#include "rcbk_table.hpp"

TEST_CASE("RcbkTable: reads the grid, data and header of mv.dat") {
    const auto t = dis::RcbkTable::read(DIS_BKFILE);

    CHECK(t.n_r() == 400);
    CHECK(t.n_Y() == 82);
    CHECK(t.N_grid.size() == 400 * 82);
    CHECK(t.MinR == 1e-6);
    CHECK(t.RMultiplier == 1.047249412298149);
    CHECK(t.x0 == 0.01);

    // Values as BKDipole reports them in Python.
    CHECK(t.r_min() == 1e-6);
    CHECK(t.r_max() == doctest::Approx(100.0000000000136).epsilon(1e-15));
    CHECK(t.Y_min() == 0.0);
    CHECK(t.Y_max() == 16.19999999999997);
    CHECK(t.Y_grid[1] == 0.2);

    // First values of the first two rapidity blocks, and the last value.
    CHECK(t.N(0, 0) == 3.962002085225981e-13);
    CHECK(t.N(0, 1) == 4.332087270474403e-13);
    CHECK(t.N(1, 0) == 4.417998064158482e-13);
    CHECK(t.N(81, 399) == 1.0);

    REQUIRE(t.Qs0_sq.has_value());
    CHECK(*t.Qs0_sq == 0.104);
    CHECK(*t.gamma == 1.0);
    CHECK(*t.ec == 1.0);
    CHECK(*t.LambdaQCD == 0.241);
}

TEST_CASE("RcbkTable: r grid is MinR * RMultiplier^i") {
    const auto t = dis::RcbkTable::read(DIS_BKFILE);
    for (std::size_t i = 1; i < t.n_r(); ++i)
        CHECK(t.r_grid[i] / t.r_grid[i - 1] == doctest::Approx(t.RMultiplier).epsilon(1e-13));
}

TEST_CASE("RcbkTable: rapidity blocks are sorted and malformed files are rejected") {
    const std::string name = "dis_test_rcbk.dat";
    auto write = [&](const std::string &body) {
        std::ofstream out(name);
        out << "# test file\n###0.1\n###2.0\n###3\n###0.01\n" << body;
    };

    // Blocks out of order: they come back sorted by Y.
    write("###1.0\n4\n5\n6\n###0.0\n1\n2\n3\n");
    auto t = dis::RcbkTable::read(name);
    CHECK(t.Y_grid == std::vector<double>{0.0, 1.0});
    CHECK(t.N_grid == std::vector<double>{1, 2, 3, 4, 5, 6});
    CHECK(t.r_grid[2] == doctest::Approx(0.4));

    // A block with too few r points.
    write("###0.0\n1\n2\n");
    CHECK_THROWS_AS(dis::RcbkTable::read(name), std::runtime_error);

    // No rapidity block at all.
    write("");
    CHECK_THROWS_AS(dis::RcbkTable::read(name), std::runtime_error);

    CHECK_THROWS_AS(dis::RcbkTable::read("does_not_exist.dat"), std::runtime_error);
    std::remove(name.c_str());
}
