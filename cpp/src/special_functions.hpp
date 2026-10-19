// Bessel functions used by the wavefunctions and the phase-space kernels.
//
// Thin wrappers around GSL, kept in one place so that a faster implementation
// can replace them without touching the physics code.
//
// GSL's default error handler aborts the program, which it would do on a
// harmless underflow such as K0(800). The wrappers use the _e variants and
// return the underflowed value (0) instead; special_functions.cpp turns the
// abort-on-error handler off when the library is loaded.

#ifndef DIS_SPECIAL_FUNCTIONS_HPP
#define DIS_SPECIAL_FUNCTIONS_HPP

namespace dis {

double bessel_K0(double x);
double bessel_K1(double x);
double bessel_J0(double x);
double bessel_J1(double x);

// Turns off GSL's abort-on-error handler. Called automatically at load time;
// exposed so that a program which installs its own handler can restore ours.
void disable_gsl_error_handler();

}  // namespace dis

#endif  // DIS_SPECIAL_FUNCTIONS_HPP
