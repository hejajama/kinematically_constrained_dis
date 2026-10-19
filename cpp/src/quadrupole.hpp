// Gaussian approximation of the quadrupole in terms of dipoles, Dominguez et
// al., Phys. Rev. D 83, 105005 (2011), eq. B21, and its large-Nc limit.
// Follows building_blocks/correlators/Quadrupoles/QuadrupoleCorrelator.py.
//
// The four positions x1, x2 (quark, antiquark in the amplitude) and x1', x2'
// (in the conjugate) make six distinct pairs; every term of the quadrupole is
// built from the dipoles of those six, so each is evaluated once.

#ifndef DIS_QUADRUPOLE_HPP
#define DIS_QUADRUPOLE_HPP

#include <cmath>

#include "constants.hpp"

namespace dis {

// Added to every dipole before its log is taken, so that log(S) stays finite.
constexpr double quadrupole_log_floor = 1e-14;

// Dipole S-matrix of each pair. Names as the keys in the Python code:
// d12 = S(x1, x2), d1p2p = S(x1', x2'), d12p = S(x1, x2'), and so on.
struct PairDipoles {
    double d12, d1p2p, d11p, d22p, d12p, d21p;
};

// Pair dipoles in polar variables, photon at the origin:
//   x1 = ((1-z) u, 0),                    x2 = (-z u, 0),
//   x1' = (1-z) up (cos theta, sin theta), x2' = -z up (cos theta, sin theta).
// `S` is a dipole at fixed rapidity (anything with double S(double r)).
template <class DipoleAtY>
PairDipoles pair_dipoles(const DipoleAtY &S, double u, double up, double z, double cos_theta,
                         double sin_theta) {
    const double c = cos_theta, s = sin_theta;
    const double x1[2] = {(1 - z) * u, 0.0};
    const double x2[2] = {-z * u, 0.0};
    const double x1p[2] = {(1 - z) * up * c, (1 - z) * up * s};
    const double x2p[2] = {-z * up * c, -z * up * s};
    auto dist = [](const double *a, const double *b) { return std::hypot(a[0] - b[0], a[1] - b[1]); };
    return {S.S(dist(x1, x2)),  S.S(dist(x2p, x1p)), S.S(dist(x1, x1p)),
            S.S(dist(x2, x2p)), S.S(dist(x1, x2p)),  S.S(dist(x2, x1p))};
}

template <class DipoleAtY>
PairDipoles pair_dipoles_polar(const DipoleAtY &S, double u, double up, double z, double theta) {
    return pair_dipoles(S, u, up, z, std::cos(theta), std::sin(theta));
}

namespace detail {

struct LogDipoles {
    double l12, l1p2p, l11p, l22p, l12p, l21p;
};

inline LogDipoles logs(const PairDipoles &S) {
    return {std::log(S.d12 + quadrupole_log_floor),  std::log(S.d1p2p + quadrupole_log_floor),
            std::log(S.d11p + quadrupole_log_floor), std::log(S.d22p + quadrupole_log_floor),
            std::log(S.d12p + quadrupole_log_floor), std::log(S.d21p + quadrupole_log_floor)};
}

// F1, F2, F3 of Dominguez et al. (below eq. B13).
struct FTerms {
    double F1, F2, F3;
};

inline FTerms F_terms(const LogDipoles &L) {
    return {(1 / CF) * (L.l12 + L.l1p2p - L.l11p - L.l22p),
            (1 / CF) * (L.l12p + L.l21p - L.l11p - L.l22p),
            (1 / CF) * (L.l12p + L.l21p - L.l12 - L.l1p2p)};
}

}  // namespace detail

// Finite-Nc quadrupole, eq. B21.
inline double quadrupole_finite_Nc(const PairDipoles &S) {
    const auto L = detail::logs(S);
    const auto [F1, F2, F3] = detail::F_terms(L);
    const double SuSup = std::exp(L.l12 + L.l1p2p);

    const double Delta = F1 * F1 + (4.0 / (Nc * Nc)) * F2 * F3;
    const double sqrt_Delta = std::sqrt(Delta);

    double big_factor;
    // A negative Delta gives NaN, which fails this test and takes the
    // degenerate branch, as in the numpy code.
    if (sqrt_Delta > 0) {
        const double term1 = ((sqrt_Delta + F1) / (2 * sqrt_Delta) - F2 / sqrt_Delta) *
                              std::exp(Nc * sqrt_Delta / 4);
        const double term2 = ((sqrt_Delta - F1) / (2 * sqrt_Delta) + F2 / sqrt_Delta) *
                              std::exp(-Nc * sqrt_Delta / 4);
        big_factor = term1 + term2;
    } else {
        // Delta = 0 limit of term1 + term2: 1 + Nc (F1/2 - F2) / 2. Reached
        // e.g. when all six dipoles are equal (a transparent target).
        big_factor = 1 + Nc * (F1 / 2 - F2) / 2;
    }
    return SuSup * big_factor * std::exp((-Nc / 4.0) * F1 + (1.0 / (2 * Nc)) * F2);
}

// Large-Nc limit of the same expression.
inline double quadrupole_large_Nc(const PairDipoles &S) {
    const auto L = detail::logs(S);
    const auto [F1, F2, F3] = detail::F_terms(L);
    (void)F3;
    const double SuSup = std::exp(L.l12 + L.l1p2p);
    const double Su_mixed = std::exp(L.l11p + L.l22p);
    return SuSup - (F2 / (F1 + 1e-12)) * (SuSup - Su_mixed);
}

// S(u), S(u') and the quadrupole S4 from six dipole evaluations. The
// finite-energy target amplitude is 1 - S(u) - S(u') + S4.
struct Correlators {
    double S_u, S_up, S4;
};

template <class DipoleAtY>
Correlators correlators(const DipoleAtY &S, double u, double up, double z, double cos_theta,
                        double sin_theta, bool largeNc) {
    const PairDipoles pairs = pair_dipoles(S, u, up, z, cos_theta, sin_theta);
    const double S4 = largeNc ? quadrupole_large_Nc(pairs) : quadrupole_finite_Nc(pairs);
    return {pairs.d12, pairs.d1p2p, S4};
}

template <class DipoleAtY>
Correlators polar_correlators(const DipoleAtY &S, double u, double up, double z, double theta,
                              bool largeNc) {
    return correlators(S, u, up, z, std::cos(theta), std::sin(theta), largeNc);
}

}  // namespace dis

#endif  // DIS_QUADRUPOLE_HPP
