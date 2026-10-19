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
#include <stdexcept>
#include <utility>
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

// Integration limits, one [lo, hi] interval per dimension.
using Limits = std::vector<std::pair<double, double>>;

namespace detail {

using CubaIntegrand = int (*)(const int *, const double *, const int *, double *, void *);

VegasResult run_vegas(int ndim, int ncomp, CubaIntegrand integrand, void *userdata,
                      const VegasSettings &settings);

template <class Integrand>
struct Mapped {
    const Integrand *f;
    std::array<double, Integrand::ndim> lo, width;
    double jacobian;
};

// Maps Cuba's unit hypercube onto the limits.
template <class Integrand>
int call_mapped(const int * /*ndim*/, const double *unit, const int * /*ncomp*/, double *out,
                void *userdata) {
    const auto &m = *static_cast<const Mapped<Integrand> *>(userdata);
    double x[Integrand::ndim];
    for (int i = 0; i < Integrand::ndim; ++i) x[i] = m.lo[i] + m.width[i] * unit[i];
    (*m.f)(x, out);
    for (int c = 0; c < Integrand::ncomp; ++c) out[c] *= m.jacobian;
    return 0;
}

}  // namespace detail

// Integrates f over the box `limits`. Integrand must provide
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
        m.lo[i] = limits[i].first;
        m.width[i] = limits[i].second - limits[i].first;
        m.jacobian *= m.width[i];
    }
    return detail::run_vegas(Integrand::ndim, Integrand::ncomp, &detail::call_mapped<Integrand>,
                             &m, settings);
}

}  // namespace dis

#endif  // DIS_VEGAS_HPP
