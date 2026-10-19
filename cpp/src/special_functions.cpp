#include "special_functions.hpp"

#ifdef DIS_BESSEL_BOOST

#include <boost/math/special_functions/bessel.hpp>

namespace dis {

namespace {

namespace bmp = boost::math::policies;

// - promote_double<false>: compute in double. Boost's default promotes to
//   long double, which costs a lot on x86 (80-bit x87) for no gain here.
// - underflow (K at large argument) returns 0, overflow and domain errors
//   (K at x <= 0) return inf/NaN and set errno instead of throwing; the
//   integrands never evaluate there.
using Policy = bmp::policy<bmp::promote_double<false>, bmp::underflow_error<bmp::ignore_error>,
                           bmp::overflow_error<bmp::errno_on_error>,
                           bmp::domain_error<bmp::errno_on_error>,
                           bmp::evaluation_error<bmp::errno_on_error>>;

}  // namespace

double bessel_K0(double x) { return boost::math::cyl_bessel_k(0, x, Policy()); }
double bessel_K1(double x) { return boost::math::cyl_bessel_k(1, x, Policy()); }
double bessel_J0(double x) { return boost::math::cyl_bessel_j(0, x, Policy()); }
double bessel_J1(double x) { return boost::math::cyl_bessel_j(1, x, Policy()); }

}  // namespace dis

#else  // GSL

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

// GSL's default error handler aborts the program, which it would do on a
// harmless underflow such as K0(800). Turn it off when the library loads.
struct DisableHandlerAtLoad {
    DisableHandlerAtLoad() { gsl_set_error_handler_off(); }
};
const DisableHandlerAtLoad disable_handler_at_load;

}  // namespace

double bessel_K0(double x) { return value_of(gsl_sf_bessel_K0_e, x); }
double bessel_K1(double x) { return value_of(gsl_sf_bessel_K1_e, x); }
double bessel_J0(double x) { return value_of(gsl_sf_bessel_J0_e, x); }
double bessel_J1(double x) { return value_of(gsl_sf_bessel_J1_e, x); }

}  // namespace dis

#endif
