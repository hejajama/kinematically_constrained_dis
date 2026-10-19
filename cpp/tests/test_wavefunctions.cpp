#include <cmath>
#include <vector>

#include "doctest.h"
#include "special_functions.hpp"
#include "test_helpers.h"
#include "wavefunctions.hpp"

namespace {

// Parameters of the reference files (defaults of test_bk_integration.py).
const dis::PhotonParameters photon{std::sqrt(2.0), 0.14, std::sqrt(2.0 / 3.0)};

}  // namespace

TEST_CASE("OT wavefunctions match Python (wf_OT.txt)") {
    const ReferenceTable ref("wf_OT.txt");
    std::vector<double> L, T;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const auto wf = dis::ot_wavefunctions_squared(photon, ref(i, "r"), ref(i, "z"));
        L.push_back(wf.L);
        T.push_back(wf.T);
    }
    // GSL and scipy implement the Bessel functions independently.
    check_columns_close("psi_L^2", L, ref.column("psi_L_sq"), 1e-12, 0);
    check_columns_close("psi_T^2", T, ref.column("psi_T_sq"), 1e-12, 0);
}

TEST_CASE("FE wavefunctions match Python (wf_FE.txt)") {
    const ReferenceTable ref("wf_FE.txt");
    std::vector<double> L, T;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const auto wf = dis::fe_wavefunctions_squared(photon, ref(i, "u"), ref(i, "up"),
                                                      ref(i, "z"), ref(i, "theta"));
        L.push_back(wf.L);
        T.push_back(wf.T);
    }
    check_columns_close("psi_L^2", L, ref.column("psi_L_sq"), 1e-12, 0);
    // T has a cos(theta) term that can cancel against the mass term, so
    // compare relative to the size of the terms rather than of the sum.
    check_columns_close("psi_T^2", T, ref.column("psi_T_sq"), 1e-12,
                        1e-14 * max_abs(ref.column("psi_T_sq")));
}

TEST_CASE("FE wavefunctions at u = up, theta = 0 equal the OT ones") {
    for (double r : {1e-4, 0.1, 1.0, 5.0})
        for (double z : {1e-6, 0.2, 0.5, 0.9}) {
            const auto ot = dis::ot_wavefunctions_squared(photon, r, z);
            const auto fe = dis::fe_wavefunctions_squared(photon, r, r, z, 0.0);
            CHECK(fe.L == doctest::Approx(ot.L).epsilon(1e-14));
            CHECK(fe.T == doctest::Approx(ot.T).epsilon(1e-14));
        }
}

TEST_CASE("Bessel functions: values and underflow") {
    CHECK(dis::bessel_K0(1.0) == doctest::Approx(0.42102443824070834).epsilon(1e-15));
    CHECK(dis::bessel_K1(1.0) == doctest::Approx(0.6019072301972346).epsilon(1e-15));
    CHECK(dis::bessel_J0(1.0) == doctest::Approx(0.7651976865579666).epsilon(1e-15));
    CHECK(dis::bessel_J1(1.0) == doctest::Approx(0.44005058574493355).epsilon(1e-15));
    // Underflows: must return 0, not abort through GSL's error handler.
    CHECK(dis::bessel_K0(1000.0) == 0.0);
    CHECK(dis::bessel_K1(1000.0) == 0.0);
}
