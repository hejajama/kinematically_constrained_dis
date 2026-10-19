// Bessel functions used by the wavefunctions and the phase-space kernels.
//
// Kept in one place so that the implementation can change without touching
// the physics code. They come from Boost.Math by default, which is 2-4x faster
// than GSL here, or from GSL when built with -Duse_boost=OFF. Both agree with
// scipy to a few ulp. A large-argument underflow of K0, K1 returns 0.

#ifndef DIS_SPECIAL_FUNCTIONS_HPP
#define DIS_SPECIAL_FUNCTIONS_HPP

namespace dis {

double bessel_K0(double x);
double bessel_K1(double x);
double bessel_J0(double x);
double bessel_J1(double x);

}  // namespace dis

#endif  // DIS_SPECIAL_FUNCTIONS_HPP
