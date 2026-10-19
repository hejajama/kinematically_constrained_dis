#include "vegas.hpp"

#include <algorithm>

#include "cuba.h"

namespace dis {

namespace detail {

namespace {

// Grid slot shared by the warm-up and the main run. Cuba has 10 slots.
constexpr int grid_slot = 1;

// One Cuba Vegas call of exactly `iterations` iterations of `points` points.
// With epsrel = epsabs = 0 the accuracy goal is never met, so the call runs
// until maxeval = iterations * points.
VegasResult vegas_call(int ndim, int ncomp, CubaIntegrand integrand, void *userdata,
                       const VegasSettings &s, long long points, int iterations, int gridno) {
    VegasResult r;
    r.value.resize(ncomp);
    r.error.resize(ncomp);
    r.chi2_prob.resize(ncomp);
    const long long total = points * iterations;
    const int flags = std::clamp(s.verbose, 0, 3);
    llVegas(ndim, ncomp, integrand, userdata, /*nvec=*/1,
            /*epsrel=*/0, /*epsabs=*/0, flags, s.seed, /*mineval=*/total, /*maxeval=*/total,
            /*nstart=*/points, /*nincrease=*/0, s.nbatch, gridno, /*statefile=*/nullptr,
            /*spin=*/nullptr, &r.neval, &r.fail, r.value.data(), r.error.data(),
            r.chi2_prob.data());
    return r;
}

}  // namespace

VegasResult run_vegas(int ndim, int ncomp, CubaIntegrand integrand, void *userdata,
                      const VegasSettings &s) {
    if (s.mcpoints < 10) throw std::invalid_argument("VegasSettings: mcpoints must be >= 10");
    if (s.iterations < 1) throw std::invalid_argument("VegasSettings: iterations must be >= 1");

    if (s.cores >= 0) {
        const int max_cores = 10000;
        cubacores(&s.cores, &max_cores);
    }

    // A negative grid number clears the slot first, so the warm-up starts
    // from a uniform grid even if an earlier integration of the same
    // dimension left one there. The trained grid is stored in the slot.
    int gridno = -grid_slot;
    if (s.warmup_iterations > 0) {
        vegas_call(ndim, ncomp, integrand, userdata, s, s.mcpoints / 10, s.warmup_iterations,
                   gridno);
        gridno = grid_slot;  // the main run starts from the trained grid
    }
    return vegas_call(ndim, ncomp, integrand, userdata, s, s.mcpoints, s.iterations, gridno);
}

}  // namespace detail

}  // namespace dis
