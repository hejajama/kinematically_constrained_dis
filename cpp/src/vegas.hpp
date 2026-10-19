// Vegas Monte Carlo integration with Cuba.
//
// Mirrors how the Python code drives vegas: a warm-up run of
// `warmup_iterations` iterations with mcpoints/10 points each trains the
// importance-sampling grid and is discarded, then `iterations` iterations of
// mcpoints points each give the result (their weighted average). The grid is
// passed from the warm-up to the main run through Cuba's grid slot `gridno`.
//
// Cuba evaluates the integrand in forked worker processes when it was built
// with fork support. Worker processes get a copy of everything the integrand
// references, so it must not rely on writing shared state.

#ifndef DIS_VEGAS_HPP
#define DIS_VEGAS_HPP

#include <array>
#include <cmath>
#include <stdexcept>
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

struct VegasResult {
    std::vector<double> value, error, chi2_prob;
    long long neval = 0;  // evaluations in the main run
    int fail = 0;         // Cuba's fail flag; nonzero is expected with epsrel = 0
};

// One integration variable: the interval [lo, hi] and how Vegas's unit
// interval t is mapped onto it, linearly (x = lo + (hi - lo) t) or
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

VegasResult run_vegas(int ndim, int ncomp, CubaIntegrand integrand, void *userdata,
                      const VegasSettings &settings);

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

}  // namespace detail

// Integrates f over `limits`. Integrand must provide
//   static constexpr int ndim, ncomp;
//   void operator()(const double *x, double *f) const;   // f[0..ncomp-1]
template <class Integrand>
VegasResult integrate_vegas(const Integrand &f, const Limits &limits,
                            const VegasSettings &settings) {
    if (static_cast<int>(limits.size()) != Integrand::ndim)
        throw std::invalid_argument("integrate_vegas: number of limits != ndim");
    detail::Mapped<Integrand> m;
    m.f = &f;
    m.jacobian = 1;
    for (int i = 0; i < Integrand::ndim; ++i) {
        const Axis &axis = limits[i];
        m.log[i] = axis.scale == Axis::Scale::log;
        if (m.log[i]) {
            if (!(axis.lo > 0 && axis.hi > axis.lo))
                throw std::invalid_argument("integrate_vegas: log axis needs 0 < lo < hi");
            m.a[i] = std::log(axis.lo);
            m.b[i] = std::log(axis.hi / axis.lo);
        } else {
            m.a[i] = axis.lo;
            m.b[i] = axis.hi - axis.lo;
            m.jacobian *= m.b[i];
        }
    }
    return detail::run_vegas(Integrand::ndim, Integrand::ncomp, &detail::call_mapped<Integrand>,
                             &m, settings);
}

}  // namespace dis

#endif  // DIS_VEGAS_HPP
