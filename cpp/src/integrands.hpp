// Integrands of the LO DIS cross sections with a BK-evolved dipole:
//
//   OTIntegrand    optical theorem, (r, z)               BK_OT_CS.py
//   FE4DIntegrand  finite energy, BK evolved in xB,
//                  (u, u', z, theta)                     BK_FE_4D_evolve_in_xB.py
//   FE5DIntegrand  finite energy, BK evolved in xP,
//                  (u, u', z, theta, M_qq^2)             BK_FE_5D_evolve_in_xP.py
//
// Each returns sigma_L and sigma_T integrands together: they differ only in
// the wavefunction, so the dipoles and the quadrupole are evaluated once.
// evaluate() takes named arguments; operator()(x, f) is the form the
// integrator calls, with f[0] = L and f[1] = T.
//
// Templates on the dipole model, see dipole.hpp.

#ifndef DIS_INTEGRANDS_HPP
#define DIS_INTEGRANDS_HPP

#include <algorithm>
#include <cmath>

#include "constants.hpp"
#include "quadrupole.hpp"
#include "special_functions.hpp"
#include "wavefunctions.hpp"

namespace dis {

struct Kinematics {
    double Q;       // GeV
    double xB;      // Bjorken x
    double mf;      // quark mass, GeV
    double Zf;      // quark charge
    double sigma0;  // GeV^-2 (sigma0/2 multiplies the dipole amplitude)
    double x0;      // initial x of the BK evolution

    PhotonParameters photon() const { return {Q, mf, Zf}; }
    // gamma* p center-of-mass energy squared, W^2 = Q^2 (1/xB - 1).
    double W2() const { return Q * Q * (1 / xB - 1); }
};

template <class Dipole>
class OTIntegrand {
public:
    static constexpr int ndim = 2;
    static constexpr int ncomp = 2;

    // The dipole is evaluated at Y = log(x0 / xB).
    OTIntegrand(const Kinematics &kin, const Dipole &dipole)
        : kin_(kin), dipole_(dipole.at(std::log(kin.x0 / kin.xB))) {}

    LT evaluate(double r, double z) const {
        const LT wf = ot_wavefunctions_squared(kin_.photon(), r, z);
        const double dipole_amp = 2 * (1 - dipole_.S(r));
        const double jac = 2 * pi * r / (z * (1 - z));
        const double common = (kin_.sigma0 / 2) * (1 / (4 * pi)) * jac;
        return {common * wf.L * dipole_amp, common * wf.T * dipole_amp};
    }

    void operator()(const double *x, double *f) const {
        const LT v = evaluate(x[0], x[1]);
        f[0] = v.L;
        f[1] = v.T;
    }

private:
    Kinematics kin_;
    typename Dipole::AtRapidity dipole_;
};

template <class Dipole>
class FE4DIntegrand {
public:
    static constexpr int ndim = 4;
    static constexpr int ncomp = 2;

    // The dipole is evaluated at Y = log(x0 / xB).
    FE4DIntegrand(const Kinematics &kin, const Dipole &dipole, bool largeNc)
        : kin_(kin), dipole_(dipole.at(std::log(kin.x0 / kin.xB))), largeNc_(largeNc) {}

    LT evaluate(double u, double up, double z, double theta) const {
        // Phase-space kernel, integrated over M_qq^2 up to Q^2 (1 - xB) / xB.
        const double Msq_max = kin_.Q * kin_.Q * (1 - kin_.xB) / kin_.xB;
        const double arg = Msq_max * z * (1 - z) - kin_.mf * kin_.mf;
        const double r2 = u * u + up * up - 2 * u * up * std::cos(theta);
        if (!(arg > 0 && r2 > 0)) return {0, 0};
        const double zeta = std::sqrt(arg * r2);
        const double I_P = zeta * bessel_J1(zeta) / (2 * pi * r2);

        const Correlators c = polar_correlators(dipole_, u, up, z, theta, largeNc_);
        const double target_amp = 1 - c.S_u - c.S_up + c.S4;

        const LT wf = fe_wavefunctions_squared(kin_.photon(), u, up, z, theta);
        const double jac = ((u * up) / (z * (1 - z))) * 2 * pi;
        const double common = (kin_.sigma0 / 2) * (1 / (4 * pi)) * jac;
        return {common * wf.L * target_amp * I_P, common * wf.T * target_amp * I_P};
    }

    void operator()(const double *x, double *f) const {
        const LT v = evaluate(x[0], x[1], x[2], x[3]);
        f[0] = v.L;
        f[1] = v.T;
    }

private:
    Kinematics kin_;
    typename Dipole::AtRapidity dipole_;
    bool largeNc_;
};

template <class Dipole>
class FE5DIntegrand {
public:
    static constexpr int ndim = 5;
    static constexpr int ncomp = 2;

    FE5DIntegrand(const Kinematics &kin, const Dipole &dipole, bool largeNc)
        : kin_(kin), dipole_(dipole), largeNc_(largeNc) {}

    LT evaluate(double u, double up, double z, double theta, double Mqq_sq) const {
        // Kinematically forbidden: M_qq^2 below the threshold mf^2 / (z (1-z)).
        const double Msq_min = kin_.mf * kin_.mf / (z * (1 - z));
        if (!(Msq_min <= Mqq_sq)) return {0, 0};

        // At the threshold itself arg can be a round-off below zero, where the
        // numpy code gives NaN; it is set to zero here.
        const double arg = z * (1 - z) * Mqq_sq - kin_.mf * kin_.mf;
        const double r2 = u * u + up * up - 2 * u * up * std::cos(theta);
        const double zeta = std::sqrt(std::max(arg * r2, 0.0));
        const double I_P = z * (1 - z) * bessel_J0(zeta) / (4 * pi);

        // BK evolved to the rapidity of the pomeron momentum fraction xP.
        const double Q2 = kin_.Q * kin_.Q;
        const double xP = (Mqq_sq + Q2) / (kin_.W2() + Q2);
        const auto dipole = dipole_.at(std::log(kin_.x0 / xP));
        const Correlators c = polar_correlators(dipole, u, up, z, theta, largeNc_);
        const double target_amp = 1 - c.S_u - c.S_up + c.S4;

        const LT wf = fe_wavefunctions_squared(kin_.photon(), u, up, z, theta);
        const double jac = ((u * up) / (z * (1 - z))) * 2 * pi;
        const double common = (kin_.sigma0 / 2) * (1 / (4 * pi)) * jac;
        return {common * wf.L * target_amp * I_P, common * wf.T * target_amp * I_P};
    }

    void operator()(const double *x, double *f) const {
        const LT v = evaluate(x[0], x[1], x[2], x[3], x[4]);
        f[0] = v.L;
        f[1] = v.T;
    }

private:
    Kinematics kin_;
    const Dipole &dipole_;
    bool largeNc_;
};

}  // namespace dis

#endif  // DIS_INTEGRANDS_HPP
