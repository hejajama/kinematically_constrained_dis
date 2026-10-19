// Integrated structure functions against the Python implementation.
//
// These run full Vegas integrations and are in the doctest suite
// "integration" (ctest -R integration). C++ and Python results are
// independent Monte Carlo estimates, so they are compared through the pull
// (difference over the combined error), except for OT, which has a
// deterministic quadrature reference.

#include <cmath>
#include <string>

#include "cross_sections.hpp"
#include "dipole.hpp"
#include "doctest.h"
#include "interpolation.hpp"
#include "test_helpers.h"

namespace {

const dis::Kinematics kin{std::sqrt(2.0), 1e-3, 0.14, std::sqrt(2.0 / 3.0), 2 * 2.57 * 18.81, 0.01};

using Dipole = dis::BKDipole<dis::BicubicSpline>;

const Dipole &mv_dipole() {
    static const Dipole dipole(DIS_BKFILE);
    return dipole;
}

dis::VegasSettings settings(long long mcpoints) {
    dis::VegasSettings s;
    s.mcpoints = mcpoints;
    s.seed = 11;
    s.nbatch = 10000;
    return s;
}

// Vegas error estimates of these oscillating integrands are not exact (two
// Python runs of FE_4D differ by 2.2 sigma), so allow 4 sigma.
constexpr double max_pull = 4.0;

void check_pull(const std::string &what, double cpp, double cpp_err, double ref, double ref_err) {
    const double pull = (cpp - ref) / std::hypot(cpp_err, ref_err);
    INFO(what << ": C++ " << cpp << " +- " << cpp_err << ", Python " << ref << " +- " << ref_err
              << ", pull " << pull);
    CHECK(std::abs(pull) < max_pull);
}

// Row of a run_python_reference.py table.
std::size_t row_of(const ReferenceTable &ref, const std::string &calc) {
    for (std::size_t i = 0; i < ref.size(); ++i)
        if (ref.text(i, "calc") == calc) return i;
    throw std::runtime_error("no row " + calc);
}

void check_against_python(const std::string &file, const std::string &calc,
                          const dis::StructureFunctions &sf) {
    const ReferenceTable ref(file);
    const std::size_t i = row_of(ref, calc);
    check_pull(calc + " FL", sf.FL, sf.FL_err, ref(i, "FL"), ref(i, "FL_err"));
    check_pull(calc + " FT", sf.FT, sf.FT_err, ref(i, "FT"), ref(i, "FT_err"));
}

}  // namespace

TEST_SUITE("integration") {
    TEST_CASE("OT structure functions match the quadrature reference") {
        const auto sf = dis::ot_structure_functions(kin, mv_dipole(), settings(1000000));
        const ReferenceTable ref("integrated_OT_quadrature.txt");
        // The quadrature error is ~1e-13, negligible.
        check_pull("OT FL", sf.FL, sf.FL_err, ref(0, "FL"), 0);
        check_pull("OT FT", sf.FT, sf.FT_err, ref(0, "FT"), 0);
        // And the Monte Carlo error is small: 1e-3 relative or better.
        CHECK(sf.FL_err < 1e-3 * sf.FL);
        CHECK(sf.FT_err < 1e-3 * sf.FT);
    }

    TEST_CASE("FE 4D structure functions match Python, finite Nc") {
        const auto sf = dis::fe4d_structure_functions(kin, mv_dipole(), false, settings(1000000));
        check_against_python("integrated_xB1e-3_mc1e7_8cores.txt", "FE_4D", sf);
    }

    TEST_CASE("FE 5D structure functions match Python, finite Nc") {
        const auto sf = dis::fe5d_structure_functions(kin, mv_dipole(), false, settings(1000000));
        check_against_python("integrated_xB1e-3_mc1e7_8cores.txt", "FE_5D", sf);
    }

    TEST_CASE("FE 4D and 5D structure functions match Python, large Nc") {
        const std::string file = "integrated_xB1e-3_mc1e6_largeNc_8cores.txt";
        check_against_python(file, "FE_4D",
                             dis::fe4d_structure_functions(kin, mv_dipole(), true, settings(1000000)));
        check_against_python(file, "FE_5D",
                             dis::fe5d_structure_functions(kin, mv_dipole(), true, settings(1000000)));
    }
}
