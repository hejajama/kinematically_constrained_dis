// Multidimensional integration with Cuba: Vegas or Cuhre.
//
// Cuhre (the default) is deterministic adaptive cubature: it subdivides the
// integration region until the estimated error is below epsrel times the
// result, or maxeval evaluations are used. Its error estimate comes from
// comparing cubature rules of different degree, so it is not statistical: it
// cannot be checked by repeating with other seeds. Check it by repeating with
// a smaller epsrel and comparing.
//
// Vegas is adaptive Monte Carlo. It mirrors how the Python code drives vegas:
// a warm-up run of `warmup_iterations` iterations with mcpoints/10 points each
// trains the importance-sampling grid and is discarded, then `iterations`
// iterations of mcpoints points each give the result (their weighted
// average). The grid is passed from the warm-up to the main run through
// Cuba's grid slot `gridno`.
//
// Both evaluate the integrand in forked worker processes when Cuba was built
// with fork support. Worker processes get a copy of everything the integrand
// references, so it must not rely on writing shared state.

#ifndef DIS_INTEGRATION_HPP
#define DIS_INTEGRATION_HPP

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace dis {

struct VegasSettings {
    long long mcpoints = 500000;  // points per iteration of the main run
    int warmup_iterations = 10;   // each with mcpoints / 10 points
    int iterations = 20;
    // 0: Sobol quasi-random numbers. Otherwise the seed of the Mersenne
    // Twister pseudo-random generator. Pseudo-random numbers give
    // statistically meaningful error estimates; Sobol often converges faster.
    int seed = 1;
    // Points sampled per batch (Cuba's nbatch). Larger batches cut the cost of
    // handing work to the worker processes: on 8 cores 10000 is 1.4-2x faster
    // than Cuba's default of 1000.
    long long nbatch = 10000;
    // Number of worker processes. -1 leaves Cuba's default (the CUBACORES
    // environment variable, otherwise the number of idle cores); 0 evaluates
    // in the calling process. This is a process-wide Cuba setting.
    int cores = -1;
    int verbose = 0;  // Cuba verbosity, 0-3
};

struct CuhreSettings {
    // Stop when the estimated error is below max(epsrel |I|, epsabs) for
    // every component, or after maxeval evaluations (then fail != 0). At
    // xB = 1e-3 on 8 cores, 1e-3 takes ~1 s per FE structure function, 1e-4
    // ~10 s; the estimates were conservative in both cases.
    double epsrel = 1e-3;
    double epsabs = 0;
    // Cuhre keeps every subregion in memory and gets slower per evaluation as
    // they accumulate; beyond a few 1e7 evaluations Vegas may be the better tool.
    long long maxeval = 50000000;
    long long mineval = 0;
    int key = 0;     // cubature rule; 0 = Cuba's default for the dimension
    int cores = -1;  // as VegasSettings::cores
    int verbose = 0;
};

enum class Integrator { cuhre, vegas };

inline Integrator integrator_from_name(const std::string &name) {
    if (name == "cuhre") return Integrator::cuhre;
    if (name == "vegas") return Integrator::vegas;
    throw std::invalid_argument("unknown integrator " + name + " (cuhre or vegas)");
}

struct IntegrationSettings {
    Integrator integrator = Integrator::cuhre;
    CuhreSettings cuhre;
    VegasSettings vegas;
};

struct IntegrationResult {
    // chi2_prob is Cuba's prob output. Vegas: the chi^2 probability of the
    // iterations being consistent (small is suspicious). Cuhre: the
    // probability that the error estimate is NOT reliable (large is suspicious).
    std::vector<double> value, error, chi2_prob;
    long long neval = 0;  // evaluations (Vegas: in the main run)
    // Cuba's fail flag. Cuhre: nonzero if epsrel was not reached within
    // maxeval. Vegas: always nonzero, since it runs to maxeval by design.
    int fail = 0;
};

// One integration variable: the interval [lo, hi] and how the integrator's
// unit interval t is mapped onto it, linearly (x = lo + (hi - lo) t) or
// logarithmically (x = lo (hi/lo)^t, for 0 < lo < hi). The Jacobian of the
// mapping is included automatically.
struct Axis {
    enum class Scale { linear, log };
    double lo, hi;
    Scale scale = Scale::linear;
};

using Limits = std::vector<Axis>;

namespace detail {

using CubaIntegrand = int (*)(const int *, const double *, const int *, double *, void *);

IntegrationResult run_vegas(int ndim, int ncomp, CubaIntegrand integrand, void *userdata,
                            const VegasSettings &settings);
IntegrationResult run_cuhre(int ndim, int ncomp, CubaIntegrand integrand, void *userdata,
                            const CuhreSettings &settings);

template <class Integrand>
struct Mapped {
    const Integrand *f;
    // Linear axes: x = a + b t. Log axes: x = exp(a + b t).
    std::array<double, Integrand::ndim> a, b;
    std::array<bool, Integrand::ndim> log;
    double jacobian;  // product of b over the linear axes
};

// Maps Cuba's unit hypercube onto the integration variables.
template <class Integrand>
int call_mapped(const int * /*ndim*/, const double *unit, const int * /*ncomp*/, double *out,
                void *userdata) {
    const auto &m = *static_cast<const Mapped<Integrand> *>(userdata);
    double x[Integrand::ndim];
    double jacobian = m.jacobian;
    for (int i = 0; i < Integrand::ndim; ++i) {
        if (m.log[i]) {
            x[i] = std::exp(m.a[i] + m.b[i] * unit[i]);
            jacobian *= m.b[i] * x[i];
        } else {
            x[i] = m.a[i] + m.b[i] * unit[i];
        }
    }
    (*m.f)(x, out);
    for (int c = 0; c < Integrand::ncomp; ++c) out[c] *= jacobian;
    return 0;
}

template <class Integrand>
Mapped<Integrand> map_limits(const Integrand &f, const Limits &limits) {
    if (static_cast<int>(limits.size()) != Integrand::ndim)
        throw std::invalid_argument("integrate: number of limits != ndim");
    Mapped<Integrand> m;
    m.f = &f;
    m.jacobian = 1;
    for (int i = 0; i < Integrand::ndim; ++i) {
        const Axis &axis = limits[i];
        m.log[i] = axis.scale == Axis::Scale::log;
        if (m.log[i]) {
            if (!(axis.lo > 0 && axis.hi > axis.lo))
                throw std::invalid_argument("integrate: log axis needs 0 < lo < hi");
            m.a[i] = std::log(axis.lo);
            m.b[i] = std::log(axis.hi / axis.lo);
        } else {
            m.a[i] = axis.lo;
            m.b[i] = axis.hi - axis.lo;
            m.jacobian *= m.b[i];
        }
    }
    return m;
}

}  // namespace detail

// Integrates f over `limits`. Integrand must provide
//   static constexpr int ndim, ncomp;
//   void operator()(const double *x, double *f) const;   // f[0..ncomp-1]
template <class Integrand>
IntegrationResult integrate_vegas(const Integrand &f, const Limits &limits,
                                  const VegasSettings &settings) {
    auto m = detail::map_limits(f, limits);
    return detail::run_vegas(Integrand::ndim, Integrand::ncomp, &detail::call_mapped<Integrand>,
                             &m, settings);
}

// As integrate_vegas. Cuhre needs at least 2 dimensions.
template <class Integrand>
IntegrationResult integrate_cuhre(const Integrand &f, const Limits &limits,
                                  const CuhreSettings &settings) {
    static_assert(Integrand::ndim >= 2, "Cuhre needs at least 2 dimensions");
    auto m = detail::map_limits(f, limits);
    return detail::run_cuhre(Integrand::ndim, Integrand::ncomp, &detail::call_mapped<Integrand>,
                             &m, settings);
}

template <class Integrand>
IntegrationResult integrate(const Integrand &f, const Limits &limits,
                            const IntegrationSettings &settings) {
    if (settings.integrator == Integrator::vegas)
        return integrate_vegas(f, limits, settings.vegas);
    return integrate_cuhre(f, limits, settings.cuhre);
}

}  // namespace dis

#endif  // DIS_INTEGRATION_HPP
