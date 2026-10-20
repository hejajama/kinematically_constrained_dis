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
// The FE integrands can be integrated in two coordinate systems for the
// dipole vectors u and u' (FECoordinates):
//   polar:    (u, u', theta), theta the angle between them, as in Python;
//   relative: (u, r, phi), r = |u - u'| and phi the angle of u - u' to u.
// The phase-space kernel is sharply peaked at u = u' (width ~ 1/M_qq), so in
// polar coordinates the peak lies along a diagonal of the integration box,
// where Vegas cannot adapt to it, and the variance is infinite. In relative
// coordinates it sits at small r. Both describe the same integral:
// d^2u d^2u' = d^2u d^2r, so du du' dtheta u u' = du dr dphi u r, i.e. the
// integrand in relative coordinates is the polar one times r/u'.
//
// Templates on the dipole model, see dipole.hpp.

#ifndef DIS_INTEGRANDS_HPP
#define DIS_INTEGRANDS_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "constants.hpp"
#include "filon.hpp"
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

enum class FECoordinates { polar, relative };

// Geometry of the dipole pair: sizes u, u', the angle theta between them, and
// r^2 = |u - u'|^2. Building it from relative coordinates keeps r^2 exact,
// where u^2 + u'^2 - 2 u u' cos(theta) would cancel catastrophically for
// r << u, which is where the FE integrands peak.
struct DipolePair {
    double u, up, cos_theta, sin_theta, r2;

    // As the Python code computes it.
    static DipolePair polar(double u, double up, double theta) {
        return {u, up, std::cos(theta), std::sin(theta), u * u + up * up - 2 * u * up * std::cos(theta)};
    }
    // u along the x axis, u' = u - r (cos phi, sin phi).
    static DipolePair relative(double u, double r, double phi) {
        const double upx = u - r * std::cos(phi), upy = -r * std::sin(phi);
        const double up = std::hypot(upx, upy);
        return {u, up, upx / up, upy / up, r * r};
    }
};

namespace detail {

// Target amplitude 1 - S(u) - S(u') + S4 of the FE integrands.
template <class DipoleAtY>
double target_amplitude(const DipoleAtY &dipole, const DipolePair &d, double z, bool largeNc) {
    const Correlators c = correlators(dipole, d.u, d.up, z, d.cos_theta, d.sin_theta, largeNc);
    return 1 - c.S_u - c.S_up + c.S4;
}

// The part of the FE integrands that does not depend on the target or on
// M_qq^2: wavefunction x jacobian x normalization.
inline LT fe_prefactor(const Kinematics &kin, const DipolePair &d, double z) {
    const LT wf = fe_wavefunctions_squared_cos(kin.photon(), d.u, d.up, z, d.cos_theta);
    const double jac = ((d.u * d.up) / (z * (1 - z))) * 2 * pi;
    const double common = (kin.sigma0 / 2) * (1 / (4 * pi)) * jac;
    return {common * wf.L, common * wf.T};
}

// Prefactor x target amplitude: the FE integrands without the phase-space kernel.
template <class DipoleAtY>
LT fe_amplitude(const Kinematics &kin, const DipoleAtY &dipole, const DipolePair &d, double z,
                bool largeNc) {
    const double target = target_amplitude(dipole, d, z, largeNc);
    const LT p = fe_prefactor(kin, d, z);
    return {p.L * target, p.T * target};
}

// Reads the dipole pair from integration variables x[0..2], in the given
// coordinates. Returns the extra jacobian factor (r/u' for relative
// coordinates), or 0 when u' falls outside [up_min, up_max], which keeps the
// integration domain the same as in polar coordinates.
inline double read_pair(FECoordinates coords, const double *x, double up_min, double up_max,
                        DipolePair &pair) {
    if (coords == FECoordinates::polar) {
        pair = DipolePair::polar(x[0], x[1], x[2]);
        return 1;
    }
    pair = DipolePair::relative(x[0], x[1], x[2]);
    if (!(pair.up >= up_min && pair.up <= up_max)) return 0;
    return x[1] / pair.up;
}

}  // namespace detail

// Integration variables of operator(): (u, u', theta, z) in polar
// coordinates, (u, r, phi, z) in relative ones. Note that z comes after the
// angle here, unlike in evaluate() (and in Python), so that the dipole
// variables are always x[0..2].
template <class Dipole>
class FE4DIntegrand {
public:
    static constexpr int ndim = 4;
    static constexpr int ncomp = 2;

    // The dipole is evaluated at Y = log(x0 / xB). In relative coordinates u'
    // is restricted to [up_min, up_max], the range u' has in polar ones.
    FE4DIntegrand(const Kinematics &kin, const Dipole &dipole, bool largeNc,
                  FECoordinates coords = FECoordinates::polar, double up_min = 0,
                  double up_max = INFINITY)
        : FE4DIntegrand(kin, dipole, std::log(kin.x0 / kin.xB), largeNc, coords, up_min, up_max) {}

    // The same integrand with the dipole at rapidity Y. With Y = log(x0), the
    // rapidity at M_qq^2 = W^2, it is the boundary term of the FE 5D cross
    // section integrated by parts (see FE5DCorrectionIntegrand).
    FE4DIntegrand(const Kinematics &kin, const Dipole &dipole, double Y, bool largeNc,
                  FECoordinates coords, double up_min, double up_max)
        : kin_(kin), dipole_(dipole.at(Y)), largeNc_(largeNc), coords_(coords), up_min_(up_min),
          up_max_(up_max) {}

    LT evaluate(const DipolePair &d, double z) const {
        // Phase-space kernel, integrated over M_qq^2 up to Q^2 (1 - xB) / xB.
        const double Msq_max = kin_.Q * kin_.Q * (1 - kin_.xB) / kin_.xB;
        const double arg = Msq_max * z * (1 - z) - kin_.mf * kin_.mf;
        if (!(arg > 0 && d.r2 > 0)) return {0, 0};
        const double zeta = std::sqrt(arg * d.r2);
        const double I_P = zeta * bessel_J1(zeta) / (2 * pi * d.r2);

        const LT a = detail::fe_amplitude(kin_, dipole_, d, z, largeNc_);
        return {a.L * I_P, a.T * I_P};
    }

    LT evaluate(double u, double up, double z, double theta) const {
        return evaluate(DipolePair::polar(u, up, theta), z);
    }

    void operator()(const double *x, double *f) const {
        f[0] = f[1] = 0;
        DipolePair pair;
        const double jac = detail::read_pair(coords_, x, up_min_, up_max_, pair);
        if (jac == 0) return;
        const LT v = evaluate(pair, x[3]);
        f[0] = jac * v.L;
        f[1] = jac * v.T;
    }

private:
    Kinematics kin_;
    typename Dipole::AtRapidity dipole_;
    bool largeNc_;
    FECoordinates coords_;
    double up_min_, up_max_;
};

// Integration variables of operator(): (u, u', theta, z, M_qq^2) in polar
// coordinates, (u, r, phi, z, M_qq^2) in relative ones.
template <class Dipole>
class FE5DIntegrand {
public:
    static constexpr int ndim = 5;
    static constexpr int ncomp = 2;

    FE5DIntegrand(const Kinematics &kin, const Dipole &dipole, bool largeNc,
                  FECoordinates coords = FECoordinates::polar, double up_min = 0,
                  double up_max = INFINITY)
        : kin_(kin), dipole_(dipole), largeNc_(largeNc), coords_(coords), up_min_(up_min),
          up_max_(up_max) {}

    LT evaluate(const DipolePair &d, double z, double Mqq_sq) const {
        // Kinematically forbidden: M_qq^2 below the threshold mf^2 / (z (1-z)).
        const double Msq_min = kin_.mf * kin_.mf / (z * (1 - z));
        if (!(Msq_min <= Mqq_sq)) return {0, 0};

        // At the threshold itself arg can be a round-off below zero, where the
        // numpy code gives NaN; it is set to zero here.
        const double arg = z * (1 - z) * Mqq_sq - kin_.mf * kin_.mf;
        const double zeta = std::sqrt(std::max(arg * d.r2, 0.0));
        const double I_P = z * (1 - z) * bessel_J0(zeta) / (4 * pi);

        // BK evolved to the rapidity of the pomeron momentum fraction xP.
        const double Q2 = kin_.Q * kin_.Q;
        const double xP = (Mqq_sq + Q2) / (kin_.W2() + Q2);
        const LT a = detail::fe_amplitude(kin_, dipole_.at(std::log(kin_.x0 / xP)), d, z, largeNc_);
        return {a.L * I_P, a.T * I_P};
    }

    LT evaluate(double u, double up, double z, double theta, double Mqq_sq) const {
        return evaluate(DipolePair::polar(u, up, theta), z, Mqq_sq);
    }

    void operator()(const double *x, double *f) const {
        f[0] = f[1] = 0;
        DipolePair pair;
        const double jac = detail::read_pair(coords_, x, up_min_, up_max_, pair);
        if (jac == 0) return;
        const LT v = evaluate(pair, x[3], x[4]);
        f[0] = jac * v.L;
        f[1] = jac * v.T;
    }

private:
    Kinematics kin_;
    const Dipole &dipole_;
    bool largeNc_;
    FECoordinates coords_;
    double up_min_, up_max_;
};

// The FE 5D cross section integrated by parts in M_qq^2.
//
// At fixed u, u', theta and z, write k^2 = z (1-z) M_qq^2 - mf^2 and A for the
// FE5DIntegrand without its phase-space kernel. The M_qq^2 integral is then
//
//   (1/4pi) int_0^{K^2} dk^2 J0(k r) A,     K^2 = z (1-z) W^2 - mf^2,
//
// and since d/dk^2 [2 k J1(k r) / r] = J0(k r), integrating by parts gives
//
//   K J1(K r) / (2 pi r) A(W^2)  -  (1/4pi) int dM^2 [2 k J1(k r) / r] dA/dM^2.
//
// The first term is the FE 4D integrand with the dipole at the rapidity of
// M_qq^2 = W^2 (FE4DIntegrand with Y = log(x0)). In the second, only the
// target amplitude T depends on M_qq^2, through Y = log(x0 / xP):
// dA/dM^2 = A/T dT/dY dY/dM^2 with dY/dM^2 = -1 / (M_qq^2 + Q^2). This class
// is that second term, including its sign:
//
//   (1/4pi) [2 k J1(k r) / r] prefactor dT/dY / (M_qq^2 + Q^2).
//
// The original form makes Monte Carlo integration hard: J0(k r) oscillates
// over the whole M_qq^2 range and its integral is a narrow peak at r = 0 that
// only emerges by cancellation. Here, dT/dY vanishes wherever the dipole does
// not depend on Y (for a BK solution, xP > x0, most of the M_qq^2 range), so
// the correction lives at small M_qq^2, where it barely oscillates.
//
// dT/dY is a central finite difference with step dY, kept inside the dipole's
// [Y_min, Y_max], so it needs nothing from the dipole but at(Y).S(r).
//
// Integration variables of operator() are those of FE5DIntegrand.
template <class Dipole>
class FE5DCorrectionIntegrand {
public:
    static constexpr int ndim = 5;
    static constexpr int ncomp = 2;

    FE5DCorrectionIntegrand(const Kinematics &kin, const Dipole &dipole, bool largeNc,
                            FECoordinates coords = FECoordinates::polar, double up_min = 0,
                            double up_max = INFINITY, double dY = 1e-4)
        : kin_(kin), dipole_(dipole), largeNc_(largeNc), coords_(coords), up_min_(up_min),
          up_max_(up_max), dY_(dY) {}

    LT evaluate(const DipolePair &d, double z, double Mqq_sq) const {
        const double Msq_min = kin_.mf * kin_.mf / (z * (1 - z));
        if (!(Msq_min <= Mqq_sq)) return {0, 0};

        const double Q2 = kin_.Q * kin_.Q;
        const double Y = std::log(kin_.x0 * (kin_.W2() + Q2) / (Mqq_sq + Q2));
        const double Y_lo = std::max(Y - dY_, dipole_.Y_min());
        const double Y_hi = std::min(Y + dY_, dipole_.Y_max());
        if (!(Y_hi > Y_lo)) return {0, 0};
        const double dT_dY = (detail::target_amplitude(dipole_.at(Y_hi), d, z, largeNc_) -
                              detail::target_amplitude(dipole_.at(Y_lo), d, z, largeNc_)) /
                             (Y_hi - Y_lo);

        // 2 k J1(k r) / r = 2 k^2 J1(zeta) / zeta, with zeta = k r; -> k^2 as r -> 0.
        const double k2 = std::max(z * (1 - z) * Mqq_sq - kin_.mf * kin_.mf, 0.0);
        const double zeta = std::sqrt(k2 * d.r2);
        const double kernel = zeta < 1e-8 ? k2 : 2 * k2 * bessel_J1(zeta) / zeta;

        const LT p = detail::fe_prefactor(kin_, d, z);
        const double factor = kernel * dT_dY / ((4 * pi) * (Mqq_sq + Q2));
        return {p.L * factor, p.T * factor};
    }

    LT evaluate(double u, double up, double z, double theta, double Mqq_sq) const {
        return evaluate(DipolePair::polar(u, up, theta), z, Mqq_sq);
    }

    void operator()(const double *x, double *f) const {
        f[0] = f[1] = 0;
        DipolePair pair;
        const double jac = detail::read_pair(coords_, x, up_min_, up_max_, pair);
        if (jac == 0) return;
        const LT v = evaluate(pair, x[3], x[4]);
        f[0] = jac * v.L;
        f[1] = jac * v.T;
    }

    // The M_qq^2 range outside which the dipole, hence T, does not depend on
    // M_qq^2 and the integrand vanishes: Y(M_qq^2) in [Y_min, Y_max].
    double Mqq_sq_lo() const { return Mqq_sq_at(dipole_.Y_max()); }
    double Mqq_sq_hi() const { return Mqq_sq_at(dipole_.Y_min()); }

private:
    double Mqq_sq_at(double Y) const {
        const double Q2 = kin_.Q * kin_.Q;
        return kin_.x0 * (kin_.W2() + Q2) * std::exp(-Y) - Q2;
    }

    Kinematics kin_;
    const Dipole &dipole_;
    bool largeNc_;
    FECoordinates coords_;
    double up_min_, up_max_;
    double dY_;
};

// The FE 5D cross section with its M_qq^2 integral done by a Filon rule.
//
// At fixed u, u', theta and z the M_qq^2 integral of FE5DIntegrand is, with
// s = k^2 = z (1-z) M_qq^2 - mf^2,
//
//   (1/4pi) int_0^{K^2} ds J0(r sqrt(s)) T(Y(s)),    K^2 = z (1-z) W^2 - mf^2,
//
// times the prefactor; only the target amplitude T depends on M_qq^2. This
// class computes that integral at each point and so is a 4D integrand, in the
// variables of FE4DIntegrand. J0 can oscillate many times over the range,
// but T is smooth, so:
//
//  - the rapidity range Y(K^2) <= Y <= Y(0) is split at the dipole's
//    breakpoints, where T may have a kink;
//  - on a piece where the dipole does not depend on Y, T is constant and the
//    integral of J0 is exact (one evaluation of T);
//  - on the other pieces T is evaluated at `nodes` Chebyshev points and
//    interpolated (barycentric formula), then integrated against J0 with a
//    cubic Filon rule on `panels` panels (see filon.hpp). Nodes and panels are
//    equally spaced in Y for xP < x0 (BK evolution is smooth in Y) and in xP
//    for xP > x0 (where an extrapolated dipole is a smooth function of x).
//
// The cost is that of `nodes` evaluations of T per varying piece, about
// 8-16 times FE4DIntegrand, and the result is as smooth in u, u', theta, z as
// the FE 4D integrand: no oscillating M_qq^2 dimension is left for the
// integrator. This does not rely on T being constant anywhere, unlike the
// integration by parts.
template <class Dipole>
class FE5DFilonIntegrand {
public:
    static constexpr int ndim = 4;
    static constexpr int ncomp = 2;

    FE5DFilonIntegrand(const Kinematics &kin, const Dipole &dipole, bool largeNc,
                       FECoordinates coords = FECoordinates::polar, double up_min = 0,
                       double up_max = INFINITY, int nodes = 8, int panels = 8)
        : kin_(kin), dipole_(dipole), largeNc_(largeNc), coords_(coords), up_min_(up_min),
          up_max_(up_max), nodes_(nodes), panels_(panels), breakpoints_(dipole.Y_breakpoints()) {
        if (nodes_ < 2 || panels_ < 1)
            throw std::invalid_argument("FE5DFilonIntegrand: needs nodes >= 2 and panels >= 1");
    }

    LT evaluate(const DipolePair &d, double z) const {
        const double zz = z * (1 - z), m2 = kin_.mf * kin_.mf, Q2 = kin_.Q * kin_.Q;
        const double W2 = kin_.W2();
        const double s_max = zz * W2 - m2;
        if (!(s_max > 0)) return {0, 0};

        const double c = kin_.x0 * (W2 + Q2);
        auto Y_of_s = [&](double s) { return std::log(c / ((s + m2) / zz + Q2)); };
        auto s_of_Y = [&](double Y) { return zz * (c * std::exp(-Y) - Q2) - m2; };
        const double r = std::sqrt(d.r2);
        auto target = [&](double Y) {
            return detail::target_amplitude(dipole_.at(Y), d, z, largeNc_);
        };

        // Pieces of the rapidity range between breakpoints; Y decreases with s.
        std::vector<double> edges = {Y_of_s(s_max), Y_of_s(0)};
        for (double b : breakpoints_)
            if (b > edges[0] && b < edges[1]) edges.push_back(b);
        std::sort(edges.begin(), edges.end());

        double integral = 0;
        std::vector<double> Y_node(nodes_), T_node(nodes_), weight(nodes_);
        for (std::size_t e = 0; e + 1 < edges.size(); ++e) {
            const double Ya = edges[e], Yb = edges[e + 1];
            if (Yb <= dipole_.Y_min() || Ya >= dipole_.Y_max()) {
                // T constant: int J0 ds exactly.
                integral += target(0.5 * (Ya + Yb)) *
                            (j0_moment_antiderivative(0, s_of_Y(Ya), r) -
                             j0_moment_antiderivative(0, s_of_Y(Yb), r));
                continue;
            }
            // The variable T is smooth in: the rapidity Y for xP < x0, where the
            // dipole comes from BK evolution in Y, and xP itself for xP > x0,
            // where a dipole is extrapolated as a function of x (and Y would
            // stretch it exponentially).
            const bool in_x = Yb <= 0;
            auto v_of_Y = [&](double Y) { return in_x ? kin_.x0 * std::exp(-Y) : Y; };
            auto Y_of_v = [&](double v) { return in_x ? std::log(kin_.x0 / v) : v; };
            const double va = v_of_Y(Ya), vb = v_of_Y(Yb);

            // Chebyshev-Lobatto nodes in v and their barycentric weights.
            const double mid = 0.5 * (va + vb), half = 0.5 * (vb - va);
            for (int k = 0; k < nodes_; ++k) {
                Y_node[k] = mid + half * std::cos(pi * k / (nodes_ - 1));  // in v
                T_node[k] = target(Y_of_v(Y_node[k]));
                weight[k] = (k % 2 ? -1.0 : 1.0) * (k == 0 || k == nodes_ - 1 ? 0.5 : 1.0);
            }
            auto interpolate = [&](double v) {
                double num = 0, den = 0;
                for (int k = 0; k < nodes_; ++k) {
                    const double diff = v - Y_node[k];
                    if (diff == 0) return T_node[k];
                    num += weight[k] / diff * T_node[k];
                    den += weight[k] / diff;
                }
                return num / den;
            };
            // Panels equally spaced in v.
            for (int p = 0; p < panels_; ++p) {
                const double s_1 = s_of_Y(Y_of_v(va + (vb - va) * p / panels_));
                const double s_2 = s_of_Y(Y_of_v(va + (vb - va) * (p + 1) / panels_));
                const double s_a = std::min(s_1, s_2), s_b = std::max(s_1, s_2);
                const auto w = filon_weights(s_a, s_b, r);
                const double s_c = 0.5 * (s_a + s_b), h = 0.5 * (s_b - s_a);
                for (int j = 0; j < 4; ++j)
                    integral += w[j] * interpolate(v_of_Y(Y_of_s(s_c + h * filon_nodes[j])));
            }
        }

        const LT pre = detail::fe_prefactor(kin_, d, z);
        const double factor = integral / (4 * pi);
        return {pre.L * factor, pre.T * factor};
    }

    LT evaluate(double u, double up, double z, double theta) const {
        return evaluate(DipolePair::polar(u, up, theta), z);
    }

    void operator()(const double *x, double *f) const {
        f[0] = f[1] = 0;
        DipolePair pair;
        const double jac = detail::read_pair(coords_, x, up_min_, up_max_, pair);
        if (jac == 0) return;
        const LT v = evaluate(pair, x[3]);
        f[0] = jac * v.L;
        f[1] = jac * v.T;
    }

private:
    Kinematics kin_;
    const Dipole &dipole_;
    bool largeNc_;
    FECoordinates coords_;
    double up_min_, up_max_;
    int nodes_, panels_;
    std::vector<double> breakpoints_;
};

}  // namespace dis

#endif  // DIS_INTEGRANDS_HPP
