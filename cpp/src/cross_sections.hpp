// Structure functions F_L, F_T (and F_2 = F_L + F_T) from the integrands,
// as computed by src/small_x_physics/Cross_Sections/InclusiveDIS/withBK/
// test_bk_integration.py: F = Q^2 / (4 pi^2 alpha_em) sigma.

#ifndef DIS_CROSS_SECTIONS_HPP
#define DIS_CROSS_SECTIONS_HPP

#include <cmath>

#include "constants.hpp"
#include "integrands.hpp"
#include "vegas.hpp"

namespace dis {

// Integration limits; the defaults are those of test_bk_integration.py.
struct IntegrationLimits {
    double r_min = 1e-6, r_max = 10.0;  // r, u and u'
    double z_min = 1e-8, z_max = 1.0 - 1e-8;
    double theta_min = 0.0, theta_max = 2 * pi;
    // M_qq^2 range of the 5D integral: [4 mf^2, W^2] in test_bk_integration.py.
    static double Mqq_sq_min(const Kinematics &k) { return 4 * k.mf * k.mf; }
    static double Mqq_sq_max(const Kinematics &k) { return k.W2(); }
};

struct StructureFunctions {
    double FL = 0, FL_err = 0, FT = 0, FT_err = 0;
    double FL_chi2_prob = 0, FT_chi2_prob = 0;
    long long neval = 0;

    double F2() const { return FL + FT; }
    double F2_err() const { return std::hypot(FL_err, FT_err); }
};

namespace detail {

inline StructureFunctions to_structure_functions(const Kinematics &k, const VegasResult &r) {
    const double prefactor = k.Q * k.Q / (4 * pi * pi * alpha_em);
    StructureFunctions sf;
    sf.FL = prefactor * r.value[0];
    sf.FL_err = prefactor * r.error[0];
    sf.FT = prefactor * r.value[1];
    sf.FT_err = prefactor * r.error[1];
    sf.FL_chi2_prob = r.chi2_prob[0];
    sf.FT_chi2_prob = r.chi2_prob[1];
    sf.neval = r.neval;
    return sf;
}

}  // namespace detail

template <class Dipole>
StructureFunctions ot_structure_functions(const Kinematics &k, const Dipole &dipole,
                                          const VegasSettings &settings,
                                          const IntegrationLimits &lim = {}) {
    const OTIntegrand<Dipole> f(k, dipole);
    const Limits limits = {{lim.r_min, lim.r_max}, {lim.z_min, lim.z_max}};
    return detail::to_structure_functions(k, integrate_vegas(f, limits, settings));
}

template <class Dipole>
StructureFunctions fe4d_structure_functions(const Kinematics &k, const Dipole &dipole,
                                            bool largeNc, const VegasSettings &settings,
                                            const IntegrationLimits &lim = {}) {
    const FE4DIntegrand<Dipole> f(k, dipole, largeNc);
    const Limits limits = {{lim.r_min, lim.r_max},
                           {lim.r_min, lim.r_max},
                           {lim.z_min, lim.z_max},
                           {lim.theta_min, lim.theta_max}};
    return detail::to_structure_functions(k, integrate_vegas(f, limits, settings));
}

template <class Dipole>
StructureFunctions fe5d_structure_functions(const Kinematics &k, const Dipole &dipole,
                                            bool largeNc, const VegasSettings &settings,
                                            const IntegrationLimits &lim = {}) {
    const FE5DIntegrand<Dipole> f(k, dipole, largeNc);
    const Limits limits = {{lim.r_min, lim.r_max},
                           {lim.r_min, lim.r_max},
                           {lim.z_min, lim.z_max},
                           {lim.theta_min, lim.theta_max},
                           {IntegrationLimits::Mqq_sq_min(k), IntegrationLimits::Mqq_sq_max(k)}};
    return detail::to_structure_functions(k, integrate_vegas(f, limits, settings));
}

}  // namespace dis

#endif  // DIS_CROSS_SECTIONS_HPP
