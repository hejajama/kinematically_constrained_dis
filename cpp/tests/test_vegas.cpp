#include <cmath>
#include <stdexcept>

#include "doctest.h"
#include "vegas.hpp"

namespace {

// prod_i exp(-x_i^2) in 4D, and a second component 1 + x_0.
struct Gaussian4D {
    static constexpr int ndim = 4;
    static constexpr int ncomp = 2;
    void operator()(const double *x, double *f) const {
        f[0] = std::exp(-(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]));
        f[1] = 1 + x[0];
    }
};

const dis::Limits box = {{-1.0, 2.0}, {-1.0, 2.0}, {-1.0, 2.0}, {-1.0, 2.0}};

dis::VegasSettings small_run(int cores) {
    dis::VegasSettings s;
    s.mcpoints = 20000;
    s.cores = cores;
    return s;
}

}  // namespace

TEST_CASE("integrate_vegas: Gaussian over a box, with the mapping Jacobian") {
    const auto r = dis::integrate_vegas(Gaussian4D{}, box, small_run(0));
    const double one_dim = std::sqrt(M_PI) / 2 * (std::erf(2.0) + std::erf(1.0));
    const double exact0 = std::pow(one_dim, 4);
    // integral of 1 + x0 over the box: 3^4 (1 + 1/2)
    const double exact1 = 81 * 1.5;

    REQUIRE(r.value.size() == 2);
    INFO("value " << r.value[0] << " +- " << r.error[0] << ", exact " << exact0);
    CHECK(std::abs(r.value[0] - exact0) < 5 * r.error[0]);
    CHECK(r.error[0] < 3e-3 * exact0);
    CHECK(r.value[1] == doctest::Approx(exact1).epsilon(1e-3));
    // Main run: 20 iterations of mcpoints points.
    CHECK(r.neval == 20 * 20000);
}

TEST_CASE("integrate_vegas: same seed reproduces, different seed differs") {
    auto s = small_run(0);
    const auto a = dis::integrate_vegas(Gaussian4D{}, box, s);
    const auto b = dis::integrate_vegas(Gaussian4D{}, box, s);
    CHECK(a.value[0] == b.value[0]);
    CHECK(a.error[0] == b.error[0]);

    s.seed = 7;
    const auto c = dis::integrate_vegas(Gaussian4D{}, box, s);
    CHECK(c.value[0] != a.value[0]);
}

TEST_CASE("integrate_vegas: worker processes give the serial result") {
    // Cuba creates its workers with fork(); a broken build of the parallel
    // path returns 0 or hangs instead of failing loudly.
    const auto serial = dis::integrate_vegas(Gaussian4D{}, box, small_run(0));
    const auto parallel = dis::integrate_vegas(Gaussian4D{}, box, small_run(2));
    CHECK(parallel.value[0] == serial.value[0]);
    CHECK(parallel.value[1] == serial.value[1]);
    CHECK(parallel.error[0] == serial.error[0]);
}

TEST_CASE("integrate_vegas: the warm-up trains the grid") {
    auto s = small_run(0);
    s.warmup_iterations = 0;
    const auto cold = dis::integrate_vegas(Gaussian4D{}, box, s);
    s.warmup_iterations = 10;
    const auto warm = dis::integrate_vegas(Gaussian4D{}, box, s);
    CHECK(warm.error[0] < cold.error[0]);
}

TEST_CASE("integrate_vegas: rejects inconsistent input") {
    const dis::Limits too_few = {{0.0, 1.0}};
    CHECK_THROWS_AS(dis::integrate_vegas(Gaussian4D{}, too_few, small_run(0)),
                    std::invalid_argument);
    auto s = small_run(0);
    s.mcpoints = 5;
    CHECK_THROWS_AS(dis::integrate_vegas(Gaussian4D{}, box, s), std::invalid_argument);
}
