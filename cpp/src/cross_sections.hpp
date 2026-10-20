// Structure functions F_L, F_T (and F_2 = F_L + F_T) from the integrands,
// as computed by src/small_x_physics/Cross_Sections/InclusiveDIS/withBK/
// test_bk_integration.py: F = Q^2 / (4 pi^2 alpha_em) sigma.
//
// How the integration variables are sampled is set by Sampling. Every choice
// computes the same integral over the same domain; they differ in how well
// the integrator can adapt to the integrand, i.e. in the error for a given number of
// points. Sampling::python() reproduces the variables of the Python code.

#ifndef DIS_CROSS_SECTIONS_HPP
#define DIS_CROSS_SECTIONS_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "constants.hpp"
#include "integrands.hpp"
#include "integration.hpp"

namespace dis {

// Integration limits; the defaults are those of test_bk_integration.py.
struct IntegrationLimits {
    double r_min = 1e-6, r_max = 10.0;  // r (OT), u and u' (FE)
    double z_min = 1e-8, z_max = 1.0 - 1e-8;
    double theta_min = 0.0, theta_max = 2 * pi;
    // M_qq^2 range of the 5D integral: [4 mf^2, W^2] in test_bk_integration.py.
    static double Mqq_sq_min(const Kinematics &k) { return 4 * k.mf * k.mf; }
    static double Mqq_sq_max(const Kinematics &k) { return k.W2(); }
};

struct Sampling {
    // Dipole sizes (r for OT, u and u' for FE) on a logarithmic scale.
    bool log_sizes = true;
    // FE: integrate over u, r = |u - u'| and the angle of u - u' (see
    // integrands.hpp) instead of u, u' and the angle between them.
    bool relative_coordinates = true;
    // With relative coordinates, r on a logarithmic scale from r_relative_min.
    // The disc r < r_relative_min is left out; the integrand is finite there,
    // so this changes the result by a relative ~ (r_relative_min M_qq)^2,
    // ~1e-15 for the default.
    bool log_relative = true;
    double r_relative_min = 1e-10;
    // FE 5D: M_qq^2 on a logarithmic scale.
    bool log_Mqq = false;
    // How the FE 5D M_qq^2 integral is done; all give the same integral.
    //  original: as the Python code, a 5D integral with the J0 kernel.
    //  by_parts: integrated by parts in M_qq^2 (FE5DCorrectionIntegrand): a 4D
    //    boundary term plus a 5D correction where the dipole depends on M_qq^2.
    //    Fast and accurate when the dipole is frozen above some x (correction at
    //    small M_qq^2 only). When it depends on x all the way to xP = 1 the
    //    correction oscillates over the whole range, and Cuhre's error estimate
    //    fails on it (results off by ~20 times the estimate at xB = 1e-3).
    //  filon: the M_qq^2 integral at each point by a Filon rule
    //    (FE5DFilonIntegrand), leaving a smooth 4D integral. Works in both
    //    cases; ~30x more expensive per point than FE 4D, but when the dipole
    //    vanishes smoothly at xP = 1 the result is smooth enough for Cuhre to
    //    converge in ~1e5 evaluations.
    //  automatic: by_parts if the dipole stops depending on Y (is frozen)
    //    inside the M_qq^2 range, filon otherwise.
    enum class FE5DMethod { automatic, original, by_parts, filon };
    FE5DMethod fe5d_method = FE5DMethod::automatic;
    // Filon: Chebyshev nodes in Y and panels per smooth piece of the M_qq^2 range.
    int filon_nodes = 8;
    int filon_panels = 8;
    double fe5d_correction_points = 0.1;
    // Use the exact symmetries z <-> 1 - z (quark <-> antiquark) and
    // angle <-> -angle (reflection): integrate z over [z_min, 1/2] and the angle
    // over [0, pi], and multiply by 4 (by 2 for OT, which has no angle).
    // Requires z_max = 1 - z_min and an angle range of [0, 2 pi].
    bool symmetries = true;

    // The variables of the Python code.
    static Sampling python() {
        Sampling s;
        s.log_sizes = s.relative_coordinates = s.log_relative = s.log_Mqq = s.symmetries = false;
        s.fe5d_method = FE5DMethod::original;
        return s;
    }
};

struct StructureFunctions {
    double FL = 0, FL_err = 0, FT = 0, FT_err = 0;
    double FL_chi2_prob = 0, FT_chi2_prob = 0;
    long long neval = 0;
    int fail = 0;  // the integrator's fail flag (see IntegrationResult)

    double F2() const { return FL + FT; }
    double F2_err() const { return std::hypot(FL_err, FT_err); }
};

namespace detail {

inline StructureFunctions to_structure_functions(const Kinematics &k, const IntegrationResult &r,
                                                 double factor) {
    const double prefactor = factor * k.Q * k.Q / (4 * pi * pi * alpha_em);
    StructureFunctions sf;
    sf.FL = prefactor * r.value[0];
    sf.FL_err = prefactor * r.error[0];
    sf.FT = prefactor * r.value[1];
    sf.FT_err = prefactor * r.error[1];
    sf.FL_chi2_prob = r.chi2_prob[0];
    sf.FT_chi2_prob = r.chi2_prob[1];
    sf.neval = r.neval;
    sf.fail = r.fail;
    return sf;
}

// Sum of two independent integrations (errors in quadrature).
inline StructureFunctions sum(const StructureFunctions &a, const StructureFunctions &b) {
    StructureFunctions sf;
    sf.FL = a.FL + b.FL;
    sf.FL_err = std::hypot(a.FL_err, b.FL_err);
    sf.FT = a.FT + b.FT;
    sf.FT_err = std::hypot(a.FT_err, b.FT_err);
    sf.FL_chi2_prob = std::min(a.FL_chi2_prob, b.FL_chi2_prob);
    sf.FT_chi2_prob = std::min(a.FT_chi2_prob, b.FT_chi2_prob);
    sf.neval = a.neval + b.neval;
    sf.fail = a.fail != 0 ? a.fail : b.fail;
    return sf;
}

inline Axis size_axis(const IntegrationLimits &lim, const Sampling &s) {
    return {lim.r_min, lim.r_max, s.log_sizes ? Axis::Scale::log : Axis::Scale::linear};
}

inline Axis z_axis(const IntegrationLimits &lim, const Sampling &s) {
    if (!s.symmetries) return {lim.z_min, lim.z_max};
    if (std::abs(lim.z_max - (1 - lim.z_min)) > 1e-15)
        throw std::invalid_argument("Sampling::symmetries needs z_max = 1 - z_min");
    return {lim.z_min, 0.5};
}

inline Axis angle_axis(const IntegrationLimits &lim, const Sampling &s) {
    if (!s.symmetries) return {lim.theta_min, lim.theta_max};
    if (lim.theta_min != 0 || std::abs(lim.theta_max - 2 * pi) > 1e-15)
        throw std::invalid_argument("Sampling::symmetries needs the angle range [0, 2 pi]");
    return {0, pi};
}

// The axes of the dipole pair, x[0..2] of the FE integrands.
inline Limits fe_pair_axes(const IntegrationLimits &lim, const Sampling &s) {
    if (!s.relative_coordinates)
        return {size_axis(lim, s), size_axis(lim, s), angle_axis(lim, s)};
    // |u - u'| <= u + u' <= 2 r_max; points with u' outside the range are
    // rejected by the integrand.
    const Axis r = s.log_relative
                       ? Axis{s.r_relative_min, 2 * lim.r_max, Axis::Scale::log}
                       : Axis{0, 2 * lim.r_max};
    return {size_axis(lim, s), r, angle_axis(lim, s)};
}

inline FECoordinates fe_coordinates(const Sampling &s) {
    return s.relative_coordinates ? FECoordinates::relative : FECoordinates::polar;
}

}  // namespace detail

template <class Dipole>
StructureFunctions ot_structure_functions(const Kinematics &k, const Dipole &dipole,
                                          const IntegrationSettings &settings,
                                          const Sampling &sampling = {},
                                          const IntegrationLimits &lim = {}) {
    const OTIntegrand<Dipole> f(k, dipole);
    const Limits limits = {detail::size_axis(lim, sampling), detail::z_axis(lim, sampling)};
    const double factor = sampling.symmetries ? 2 : 1;
    return detail::to_structure_functions(k, integrate(f, limits, settings), factor);
}

template <class Dipole>
StructureFunctions fe4d_structure_functions(const Kinematics &k, const Dipole &dipole,
                                            bool largeNc, const IntegrationSettings &settings,
                                            const Sampling &sampling = {},
                                            const IntegrationLimits &lim = {}) {
    const FE4DIntegrand<Dipole> f(k, dipole, largeNc, detail::fe_coordinates(sampling), lim.r_min,
                                  lim.r_max);
    Limits limits = detail::fe_pair_axes(lim, sampling);
    limits.push_back(detail::z_axis(lim, sampling));
    const double factor = sampling.symmetries ? 4 : 1;
    return detail::to_structure_functions(k, integrate(f, limits, settings), factor);
}

// True if every dipole at this rapidity is 1 (a transparent target), checked
// on sizes covering the integration range.
template <class DipoleAtY>
bool transparent(const DipoleAtY &dipole, const IntegrationLimits &lim) {
    for (int i = 0; i <= 64; ++i) {
        const double r = lim.r_min * std::pow(2 * lim.r_max / lim.r_min, i / 64.0);
        if (dipole.S(r) != 1.0) return false;
    }
    return true;
}

template <class Dipole>
StructureFunctions fe5d_structure_functions(const Kinematics &k, const Dipole &dipole,
                                            bool largeNc, const IntegrationSettings &settings,
                                            const Sampling &sampling = {},
                                            const IntegrationLimits &lim = {}) {
    using Method = Sampling::FE5DMethod;
    const auto coords = detail::fe_coordinates(sampling);
    const double factor = sampling.symmetries ? 4 : 1;
    const auto M_scale = sampling.log_Mqq ? Axis::Scale::log : Axis::Scale::linear;
    Limits limits = detail::fe_pair_axes(lim, sampling);
    limits.push_back(detail::z_axis(lim, sampling));

    // Rapidity at M_qq^2 = W^2 (xP = 1).
    const double Q2 = k.Q * k.Q;
    const double Y_at_W2 = std::log(k.x0 * (k.W2() + Q2) / (IntegrationLimits::Mqq_sq_max(k) + Q2));
    Method method = sampling.fe5d_method;
    if (method == Method::automatic)
        method = dipole.Y_min() > Y_at_W2 ? Method::by_parts : Method::filon;

    if (method == Method::filon) {
        const FE5DFilonIntegrand<Dipole> f(k, dipole, largeNc, coords, lim.r_min, lim.r_max,
                                           sampling.filon_nodes, sampling.filon_panels);
        return detail::to_structure_functions(k, integrate(f, limits, settings), factor);
    }

    if (method == Method::original) {
        const FE5DIntegrand<Dipole> f(k, dipole, largeNc, coords, lim.r_min, lim.r_max);
        limits.push_back(
            {IntegrationLimits::Mqq_sq_min(k), IntegrationLimits::Mqq_sq_max(k), M_scale});
        return detail::to_structure_functions(k, integrate(f, limits, settings), factor);
    }

    // By parts. Boundary term: the FE 4D integrand at the rapidity of
    // M_qq^2 = W^2. It vanishes if the target is transparent there (as with
    // LargeXExtrapolation, where xP = 1 means N = 0); integrating zero would
    // only waste Cuhre's evaluations on an unreachable relative accuracy.
    StructureFunctions sf_boundary;
    const bool with_boundary = !transparent(dipole.at(Y_at_W2), lim);
    if (with_boundary) {
        const FE4DIntegrand<Dipole> boundary(k, dipole, Y_at_W2, largeNc, coords, lim.r_min,
                                             lim.r_max);
        sf_boundary =
            detail::to_structure_functions(k, integrate(boundary, limits, settings), factor);
    }

    // Correction, over the part of the M_qq^2 range where the dipole depends on it.
    const FE5DCorrectionIntegrand<Dipole> correction(k, dipole, largeNc, coords, lim.r_min,
                                                     lim.r_max);
    const double M_lo = std::max(IntegrationLimits::Mqq_sq_min(k), correction.Mqq_sq_lo());
    const double M_hi = std::min(IntegrationLimits::Mqq_sq_max(k), correction.Mqq_sq_hi());
    if (!(M_hi > M_lo)) return sf_boundary;
    limits.push_back({M_lo, M_hi, M_scale});
    IntegrationSettings correction_settings = settings;
    // Without a boundary term the correction is the whole result.
    const double fraction = with_boundary ? sampling.fe5d_correction_points : 1.0;
    correction_settings.vegas.mcpoints =
        std::max(1000LL, static_cast<long long>(fraction * settings.vegas.mcpoints));
    const StructureFunctions sf_correction = detail::to_structure_functions(
        k, integrate(correction, limits, correction_settings), factor);
    return detail::sum(sf_boundary, sf_correction);
}

}  // namespace dis

#endif  // DIS_CROSS_SECTIONS_HPP
