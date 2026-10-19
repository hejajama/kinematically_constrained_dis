#include "special_functions.hpp"

#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_bessel.h>

namespace dis {

namespace {

// Evaluates a GSL _e function and returns its value. An underflow leaves the
// value at 0, which is the correct answer for the decaying K functions.
template <class Function>
double value_of(Function f, double x) {
    gsl_sf_result result;
    f(x, &result);
    return result.val;
}

struct DisableHandlerAtLoad {
    DisableHandlerAtLoad() { disable_gsl_error_handler(); }
};
const DisableHandlerAtLoad disable_handler_at_load;

}  // namespace

void disable_gsl_error_handler() { gsl_set_error_handler_off(); }

double bessel_K0(double x) { return value_of(gsl_sf_bessel_K0_e, x); }
double bessel_K1(double x) { return value_of(gsl_sf_bessel_K1_e, x); }
double bessel_J0(double x) { return value_of(gsl_sf_bessel_J0_e, x); }
double bessel_J1(double x) { return value_of(gsl_sf_bessel_J1_e, x); }

}  // namespace dis
