// Filon-type quadrature of integrals int ds J0(r sqrt(s)) T(s) with smooth T.
//
// J0(r sqrt(s)) oscillates over the integration range when r is large, so an
// ordinary quadrature rule needs points for every oscillation. A Filon rule
// instead interpolates only T by a polynomial and integrates J0 times the
// polynomial exactly, through the moments int s^n J0(r sqrt(s)) ds. Its
// accuracy is then that of interpolating T, however much J0 oscillates.
//
// Here the interpolant is cubic on each panel, through the four points
// t = -1, -1/2, 1/2, 1 of the panel's local variable t = (s - s_c) / h.

#ifndef DIS_FILON_HPP
#define DIS_FILON_HPP

#include <array>

namespace dis {

// int_0^s t^n J0(r sqrt(t)) dt, for n = 0..3 and s, r >= 0.
double j0_moment_antiderivative(int n, double s, double r);

// The same for n = 0, 1, 2, 3 at once (one evaluation of J0 and J1).
std::array<double, 4> j0_moment_antiderivatives(double s, double r);

// The local nodes of a panel, t = (s - s_c) / h.
constexpr std::array<double, 4> filon_nodes = {-1.0, -0.5, 0.5, 1.0};

// Weights w_j such that int_{s_a}^{s_b} J0(r sqrt(s)) p(s) ds = sum_j w_j p(s_j)
// exactly for every cubic p, with s_j = s_c + h filon_nodes[j],
// s_c = (s_a + s_b)/2 and h = (s_b - s_a)/2.
std::array<double, 4> filon_weights(double s_a, double s_b, double r);

}  // namespace dis

#endif  // DIS_FILON_HPP
