#include <cmath>
#include <stdexcept>

#include "doctest.h"
#include "integration.hpp"

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

struct InverseX {
    static constexpr int ndim = 1;
    static constexpr int ncomp = 1;
    void operator()(const double *x, double *f) const { f[0] = 1 / x[0]; }
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

TEST_CASE("integrate_vegas: log axes include the jacobian of the mapping") {
    // 1/x on a log axis is constant in the sampling variable. Vegas does not
    // integrate even a constant exactly (it refines its grid from noisy
    // estimates, which makes the weights vary), but the variance is tiny.
    const dis::Limits log_axis = {{1e-3, 10.0, dis::Axis::Scale::log}};
    const auto r = dis::integrate_vegas(InverseX{}, log_axis, small_run(0));
    CHECK(std::abs(r.value[0] - std::log(1e4)) < 5 * r.error[0]);
    CHECK(r.error[0] < 1e-4 * std::log(1e4));

    const dis::Limits bad = {{0.0, 10.0, dis::Axis::Scale::log}};
    CHECK_THROWS_AS(dis::integrate_vegas(InverseX{}, bad, small_run(0)), std::invalid_argument);
}

TEST_CASE("integrate_vegas: rejects inconsistent input") {
    const dis::Limits too_few = {{0.0, 1.0}};
    CHECK_THROWS_AS(dis::integrate_vegas(Gaussian4D{}, too_few, small_run(0)),
                    std::invalid_argument);
    auto s = small_run(0);
    s.mcpoints = 5;
    CHECK_THROWS_AS(dis::integrate_vegas(Gaussian4D{}, box, s), std::invalid_argument);
}

TEST_CASE("integrate_cuhre: Gaussian over a box to the requested accuracy") {
    dis::CuhreSettings s;
    s.epsrel = 1e-8;
    s.cores = 0;
    const auto r = dis::integrate_cuhre(Gaussian4D{}, box, s);
    const double one_dim = std::sqrt(M_PI) / 2 * (std::erf(2.0) + std::erf(1.0));
    const double exact0 = std::pow(one_dim, 4);
    CHECK(r.fail == 0);
    CHECK(r.error[0] <= 1e-8 * exact0);
    CHECK(r.value[0] == doctest::Approx(exact0).epsilon(1e-8));
    CHECK(r.value[1] == doctest::Approx(81 * 1.5).epsilon(1e-12));
}

TEST_CASE("integrate_cuhre: deterministic, and worker processes give the serial result") {
    dis::CuhreSettings s;
    s.epsrel = 1e-6;
    s.cores = 0;
    const auto serial = dis::integrate_cuhre(Gaussian4D{}, box, s);
    s.cores = 2;
    const auto parallel = dis::integrate_cuhre(Gaussian4D{}, box, s);
    CHECK(parallel.value[0] == serial.value[0]);
    CHECK(parallel.error[0] == serial.error[0]);
    CHECK(parallel.neval == serial.neval);
}

TEST_CASE("integrate_cuhre: reports failure when maxeval is too small") {
    dis::CuhreSettings s;
    s.epsrel = 1e-12;
    s.maxeval = 5000;
    s.cores = 0;
    const auto r = dis::integrate_cuhre(Gaussian4D{}, box, s);
    CHECK(r.fail != 0);
    CHECK(r.neval <= 5000 + 2000);  // stops after the region that crosses maxeval
}

TEST_CASE("integrate dispatches to the selected integrator") {
    dis::IntegrationSettings s;
    CHECK(s.integrator == dis::Integrator::cuhre);
    s.cuhre.cores = s.vegas.cores = 0;
    s.vegas.mcpoints = 20000;
    const auto c = dis::integrate(Gaussian4D{}, box, s);
    CHECK(c.value[0] == dis::integrate_cuhre(Gaussian4D{}, box, s.cuhre).value[0]);
    s.integrator = dis::Integrator::vegas;
    const auto v = dis::integrate(Gaussian4D{}, box, s);
    CHECK(v.value[0] == dis::integrate_vegas(Gaussian4D{}, box, s.vegas).value[0]);

    CHECK(dis::integrator_from_name("vegas") == dis::Integrator::vegas);
    CHECK(dis::integrator_from_name("cuhre") == dis::Integrator::cuhre);
    CHECK_THROWS_AS(dis::integrator_from_name("suave"), std::invalid_argument);
}
