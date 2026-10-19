// Leading-order squared photon light-cone wavefunctions, longitudinal (L) and
// transverse (T), for one quark flavor.
//
// OT: optical theorem, building_blocks/wavefunctions/OT_photon_wavefunctions/LO.py
// FE: finite-energy constrained, building_blocks/wavefunctions/FE_photon_wavefunctions/LO.py
//
// L and T are returned together because they share the Bessel functions.

#ifndef DIS_WAVEFUNCTIONS_HPP
#define DIS_WAVEFUNCTIONS_HPP

#include <algorithm>
#include <cmath>

#include "constants.hpp"
#include "special_functions.hpp"

namespace dis {

struct LT {
    double L = 0;
    double T = 0;
};

struct PhotonParameters {
    double Q;   // photon virtuality, GeV
    double mf;  // quark mass, GeV
    double Zf;  // quark charge, units of e
};

namespace detail {

// epsilon^2 = Q^2 z (1 - z) + mf^2, floored as in the Python code.
inline double epsilon_sq(const PhotonParameters &p, double z) {
    return std::max(p.Q * p.Q * z * (1 - z) + p.mf * p.mf, 1e-12);
}

}  // namespace detail

// OT wavefunctions squared at dipole size r and momentum fraction z.
inline LT ot_wavefunctions_squared(const PhotonParameters &p, double r, double z) {
    const double Cf_T = 2 * Nc * alpha_em * p.Zf * p.Zf / pi;
    const double Cf_L = 8 * Nc * alpha_em * p.Zf * p.Zf / pi;
    const double eps2 = detail::epsilon_sq(p, z);
    const double x = r * std::sqrt(eps2);
    const double K0 = bessel_K0(x);
    const double K1 = bessel_K1(x);

    const double term1 = (z * z + (1 - z) * (1 - z)) * eps2 * (K1 * K1);
    const double term2 = p.mf * p.mf * (K0 * K0);
    const double zbar = 1 - z;
    return {Cf_L * p.Q * p.Q * (z * z * z) * (zbar * zbar * zbar) * (K0 * K0),
            Cf_T * z * zbar * (term1 + term2)};
}

// FE wavefunction times conjugate, with dipole sizes u (amplitude) and up
// (conjugate amplitude) at relative angle theta.
inline LT fe_wavefunctions_squared(const PhotonParameters &p, double u, double up, double z,
                                   double theta) {
    const double Cf_T = 2 * Nc * alpha_em * p.Zf * p.Zf / pi;
    const double Cf_L = 8 * Nc * alpha_em * p.Zf * p.Zf / pi;
    const double eps2 = detail::epsilon_sq(p, z);
    const double sqrt_eps = std::sqrt(eps2);
    const double K0_sq = bessel_K0(u * sqrt_eps) * bessel_K0(up * sqrt_eps);
    const double K1_sq = bessel_K1(u * sqrt_eps) * bessel_K1(up * sqrt_eps);

    const double term1 = (z * z + (1 - z) * (1 - z)) * eps2 * std::cos(theta) * K1_sq;
    const double term2 = p.mf * p.mf * K0_sq;
    const double zbar = 1 - z;
    return {Cf_L * p.Q * p.Q * (z * z * z) * (zbar * zbar * zbar) * K0_sq,
            Cf_T * z * zbar * (term1 + term2)};
}

}  // namespace dis

#endif  // DIS_WAVEFUNCTIONS_HPP
