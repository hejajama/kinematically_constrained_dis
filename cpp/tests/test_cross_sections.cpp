// Integrated structure functions against the Python implementation.
//
// These run full integrations, with Cuhre and with Vegas, and are in the
// doctest suite "integration" (ctest -R integration). They are compared with
// the Python results through the pull (difference over the combined error),
// and OT also with a deterministic quadrature reference.

#include <algorithm>
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

dis::IntegrationSettings vegas(long long mcpoints) {
    dis::IntegrationSettings s;
    s.integrator = dis::Integrator::vegas;
    s.vegas.mcpoints = mcpoints;
    s.vegas.seed = 11;
    return s;
}

dis::IntegrationSettings cuhre(double epsrel) {
    dis::IntegrationSettings s;
    s.integrator = dis::Integrator::cuhre;
    s.cuhre.epsrel = epsrel;
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

// FE_5D from three Python runs at 1e7 points with different seeds. Their
// scatter is about twice the error vegas reports for one run (its integrand
// in these variables has infinite variance), so the error of the mean is
// taken as the larger of the two estimates.
void check_against_python_fe5d(const dis::StructureFunctions &sf) {
    const ReferenceTable runs[] = {ReferenceTable("integrated_xB1e-3_mc1e7_8cores.txt"),
                                   ReferenceTable("integrated_xB1e-3_mc1e7_FE5D_seed5.txt"),
                                   ReferenceTable("integrated_xB1e-3_mc1e7_FE5D_seed6.txt")};
    for (const std::string F : {"FL", "FT"}) {
        double mean = 0, reported = 0, scatter = 0;
        for (const auto &r : runs) {
            mean += r(row_of(r, "FE_5D"), F) / 3;
            reported += r(row_of(r, "FE_5D"), F + "_err") / 3;
        }
        for (const auto &r : runs) scatter += std::pow(r(row_of(r, "FE_5D"), F) - mean, 2) / 2;
        const double err = std::max(reported, std::sqrt(scatter)) / std::sqrt(3.0);
        check_pull("FE_5D " + F, F == "FL" ? sf.FL : sf.FT, F == "FL" ? sf.FL_err : sf.FT_err,
                   mean, err);
    }
}

}  // namespace

TEST_SUITE("integration") {
    // Cuhre, the default integrator, with the default sampling.

    TEST_CASE("Cuhre: OT structure functions match the quadrature reference") {
        const auto sf = dis::ot_structure_functions(kin, mv_dipole(), cuhre(1e-7));
        const ReferenceTable ref("integrated_OT_quadrature.txt");
        CHECK(sf.fail == 0);
        CHECK(sf.FL == doctest::Approx(ref(0, "FL")).epsilon(1e-7));
        CHECK(sf.FT == doctest::Approx(ref(0, "FT")).epsilon(1e-7));
        CHECK(std::abs(sf.FL - ref(0, "FL")) <= sf.FL_err);
        CHECK(std::abs(sf.FT - ref(0, "FT")) <= sf.FT_err);
    }

    TEST_CASE("Cuhre: FE 4D and 5D structure functions match Python") {
        const auto fe4d = dis::fe4d_structure_functions(kin, mv_dipole(), false, cuhre(1e-3));
        CHECK(fe4d.fail == 0);
        check_against_python("integrated_xB1e-3_mc1e7_8cores.txt", "FE_4D", fe4d);
        const auto fe5d = dis::fe5d_structure_functions(kin, mv_dipole(), false, cuhre(1e-3));
        CHECK(fe5d.fail == 0);
        check_against_python_fe5d(fe5d);

        const std::string file = "integrated_xB1e-3_mc1e6_largeNc_8cores.txt";
        check_against_python(file, "FE_4D",
                             dis::fe4d_structure_functions(kin, mv_dipole(), true, cuhre(1e-3)));
        check_against_python(file, "FE_5D",
                             dis::fe5d_structure_functions(kin, mv_dipole(), true, cuhre(1e-3)));
    }

    TEST_CASE("Cuhre: the error estimate covers the change to a 10x tighter epsrel") {
        // Cuhre's error estimate is not statistical; check it against a more
        // accurate result instead.
        const auto coarse = dis::fe5d_structure_functions(kin, mv_dipole(), false, cuhre(1e-3));
        const auto fine = dis::fe5d_structure_functions(kin, mv_dipole(), false, cuhre(1e-4));
        CHECK(std::abs(coarse.FL - fine.FL) <= coarse.FL_err);
        CHECK(std::abs(coarse.FT - fine.FT) <= coarse.FT_err);
        CHECK(fine.FL_err < 0.2 * coarse.FL_err);
    }

    // Vegas with the default sampling.

    TEST_CASE("Vegas: OT structure functions match the quadrature reference") {
        const auto sf = dis::ot_structure_functions(kin, mv_dipole(), vegas(1000000));
        const ReferenceTable ref("integrated_OT_quadrature.txt");
        // The quadrature error is ~1e-13, negligible.
        check_pull("OT FL", sf.FL, sf.FL_err, ref(0, "FL"), 0);
        check_pull("OT FT", sf.FT, sf.FT_err, ref(0, "FT"), 0);
        // And the Monte Carlo error is small: 1e-3 relative or better.
        CHECK(sf.FL_err < 1e-3 * sf.FL);
        CHECK(sf.FT_err < 1e-3 * sf.FT);
    }

    TEST_CASE("Vegas: FE 4D structure functions match Python, finite Nc") {
        const auto sf = dis::fe4d_structure_functions(kin, mv_dipole(), false, vegas(1000000));
        check_against_python("integrated_xB1e-3_mc1e7_8cores.txt", "FE_4D", sf);
        CHECK(sf.FL_err < 3e-3 * sf.FL);
    }

    TEST_CASE("Vegas: FE 5D structure functions match Python, finite Nc") {
        const auto sf = dis::fe5d_structure_functions(kin, mv_dipole(), false, vegas(1000000));
        check_against_python_fe5d(sf);
        // Integrated by parts, the error is far below that of the original form.
        CHECK(sf.FL_err < 3e-3 * sf.FL);
        CHECK(sf.FT_err < 3e-3 * sf.FT);
    }

    TEST_CASE("Vegas: FE 4D and 5D structure functions match Python, large Nc") {
        const std::string file = "integrated_xB1e-3_mc1e6_largeNc_8cores.txt";
        check_against_python(file, "FE_4D",
                             dis::fe4d_structure_functions(kin, mv_dipole(), true, vegas(1000000)));
        check_against_python(file, "FE_5D",
                             dis::fe5d_structure_functions(kin, mv_dipole(), true, vegas(1000000)));
    }

    // Vegas with the variables of the Python code.

    TEST_CASE("Vegas, Python variables: FE 4D and 5D structure functions match Python") {
        const auto python = dis::Sampling::python();
        check_against_python(
            "integrated_xB1e-3_mc1e7_8cores.txt", "FE_4D",
            dis::fe4d_structure_functions(kin, mv_dipole(), false, vegas(1000000), python));
        check_against_python_fe5d(
            dis::fe5d_structure_functions(kin, mv_dipole(), false, vegas(1000000), python));
    }
}
