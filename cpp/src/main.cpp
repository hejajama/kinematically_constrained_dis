// Leading-order DIS structure functions with a BK-evolved dipole: optical
// theorem (OT) and finite-energy constrained, BK evolved in xB (FE_4D) or in
// xP (FE_5D).
//
// Command line and output follow
// src/small_x_physics/Cross_Sections/InclusiveDIS/withBK/test_bk_integration.py,
// so the two can be compared directly. Timings go to stderr.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

#include "cross_sections.hpp"
#include "dipole.hpp"
#include "interpolation.hpp"

namespace {

struct Options {
    dis::Kinematics kin{std::sqrt(2.0), 1e-2, 0.14, std::sqrt(2.0 / 3.0), 2 * 2.57 * 18.81, 0.01};
    std::string bkfile;
    bool largeNc = false;
    std::string interpolation = dis::BicubicSpline::name;
    std::set<std::string> only;  // empty: all
    dis::IntegrationSettings integration;
    dis::Sampling sampling;
};

void usage(const char *prog) {
    std::cerr
        << "Usage: " << prog << " --bkfile FILE [options]\n"
        << "\n"
        << "Physics (defaults as test_bk_integration.py):\n"
        << "  --Q Q            photon virtuality, GeV (sqrt(2))\n"
        << "  --xB X           Bjorken x (1e-2)\n"
        << "  --mf M           quark mass, GeV (0.14)\n"
        << "  --Zf Z           quark charge (sqrt(2/3))\n"
        << "  --sigma0 S       GeV^-2 (2*2.57*18.81)\n"
        << "  --x0 X           initial x of the BK evolution (0.01)\n"
        << "  --largeNc        large-Nc quadrupole (FE rows only)\n"
        << "  --Qs0, --gamma, --ec   accepted for compatibility, unused (taken from the BK file)\n"
        << "\n"
        << "Numerics:\n"
        << "  --integrator NAME   cuhre (default, deterministic cubature) or vegas (Monte Carlo)\n"
        << "  --interpolation NAME   bicubic (default, as the Python code) or bilinear\n"
        << "  --only CALC      OT, FE_4D or FE_5D; may be repeated (default: all)\n"
        << "  --cores N        worker processes, 0 = none (default: CUBACORES or idle cores)\n"
        << "  --verbose V      Cuba verbosity 0-3 (0)\n"
        << "\n"
        << "Cuhre:\n"
        << "  --epsrel E       target relative error (1e-3)\n"
        << "  --maxeval N      at most N integrand evaluations per integral (5e7)\n"
        << "\n"
        << "Vegas:\n"
        << "  --mcpoints N     points per iteration (500000)\n"
        << "  --seed S         random seed, 0 = Sobol quasi-random (1)\n"
        << "  --nbatch N       batch size (10000)\n"
        << "  --iterations N   main-run iterations (20)\n"
        << "  --warmup N       warm-up iterations with mcpoints/10 points (10)\n"
        << "\n"
        << "Sampling (see cross_sections.hpp; all compute the same integral):\n"
        << "  --variables python|optimized   python: the variables of the Python code;\n"
        << "                   optimized (default): the settings below\n"
        << "  --linear-sizes   dipole sizes on a linear scale (default: log)\n"
        << "  --polar          FE in (u, u', theta) (default: (u, |u-u'|, phi))\n"
        << "  --linear-relative   |u-u'| on a linear scale (default: log)\n"
        << "  --log-Mqq        FE 5D: M_qq^2 on a log scale (default: linear)\n"
        << "  --no-by-parts    FE 5D: integrate the original form, not by parts in M_qq^2\n"
        << "  --no-symmetries  do not use z <-> 1-z and angle <-> -angle\n";
}

Options parse(int argc, char **argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "--Q") o.kin.Q = std::stod(value());
        else if (arg == "--xB") o.kin.xB = std::stod(value());
        else if (arg == "--mf") o.kin.mf = std::stod(value());
        else if (arg == "--Zf") o.kin.Zf = std::stod(value());
        else if (arg == "--sigma0") o.kin.sigma0 = std::stod(value());
        else if (arg == "--x0") o.kin.x0 = std::stod(value());
        else if (arg == "--Qs0" || arg == "--gamma" || arg == "--ec") value();
        else if (arg == "--bkfile") o.bkfile = value();
        else if (arg == "--largeNc") o.largeNc = true;
        else if (arg == "--integrator")
            o.integration.integrator = dis::integrator_from_name(value());
        else if (arg == "--epsrel") o.integration.cuhre.epsrel = std::stod(value());
        else if (arg == "--maxeval") o.integration.cuhre.maxeval = std::llround(std::stod(value()));
        else if (arg == "--mcpoints") o.integration.vegas.mcpoints = std::stoll(value());
        else if (arg == "--interpolation") o.interpolation = value();
        else if (arg == "--only") o.only.insert(value());
        else if (arg == "--seed") o.integration.vegas.seed = std::stoi(value());
        else if (arg == "--cores")
            o.integration.vegas.cores = o.integration.cuhre.cores = std::stoi(value());
        else if (arg == "--nbatch") o.integration.vegas.nbatch = std::stoll(value());
        else if (arg == "--iterations") o.integration.vegas.iterations = std::stoi(value());
        else if (arg == "--warmup") o.integration.vegas.warmup_iterations = std::stoi(value());
        else if (arg == "--verbose")
            o.integration.vegas.verbose = o.integration.cuhre.verbose = std::stoi(value());
        else if (arg == "--variables") {
            const std::string v = value();
            if (v == "python") o.sampling = dis::Sampling::python();
            else if (v == "optimized") o.sampling = dis::Sampling{};
            else throw std::invalid_argument("--variables must be python or optimized");
        }
        else if (arg == "--linear-sizes") o.sampling.log_sizes = false;
        else if (arg == "--polar") o.sampling.relative_coordinates = false;
        else if (arg == "--linear-relative") o.sampling.log_relative = false;
        else if (arg == "--log-Mqq") o.sampling.log_Mqq = true;
        else if (arg == "--no-by-parts") o.sampling.fe5d_by_parts = false;
        else if (arg == "--no-symmetries") o.sampling.symmetries = false;
        else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else throw std::invalid_argument("unknown option " + arg);
    }
    if (o.bkfile.empty()) throw std::invalid_argument("--bkfile is required");
    for (const auto &calc : o.only)
        if (calc != "OT" && calc != "FE_4D" && calc != "FE_5D")
            throw std::invalid_argument("--only must be OT, FE_4D or FE_5D, got " + calc);
    return o;
}

template <class Interpolator>
void run(const Options &o) {
    const dis::BKDipole<Interpolator> dipole(o.bkfile);
    const auto &table = dipole.table();
    const double Y_fixed = std::log(o.kin.x0 / o.kin.xB);
    if (Y_fixed < table.Y_min() || Y_fixed > table.Y_max())
        std::cerr << "Warning: Y = log(x0/xB) = " << Y_fixed << " is outside the tabulated range ["
                  << table.Y_min() << ", " << table.Y_max() << "] of " << o.bkfile << "\n";

    // Same columns as test_bk_integration.py. With Cuhre, which has no
    // mcpoints, that column holds the number of integrand evaluations.
    const bool vegas = o.integration.integrator == dis::Integrator::vegas;
    std::printf("calc, Q2, xB, m, largeNc, mcpoints, FL, FL_err, FT, FT_err, F2, F2_err\n");
    auto run_one = [&](const char *calc, auto compute) {
        if (!o.only.empty() && !o.only.count(calc)) return;
        const auto start = std::chrono::steady_clock::now();
        const dis::StructureFunctions sf = compute();
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
        std::printf("%s, %.17g, %.17g, %.17g, %d, %lld, %.17g, %.17g, %.17g, %.17g, %.17g, %.17g\n",
                    calc, o.kin.Q * o.kin.Q, o.kin.xB, o.kin.mf, int(o.largeNc),
                    vegas ? o.integration.vegas.mcpoints : sf.neval, sf.FL, sf.FL_err, sf.FT,
                    sf.FT_err, sf.F2(), sf.F2_err());
        std::fflush(stdout);
        std::fprintf(stderr, "%s: %.3f s, %lld evaluations", calc, elapsed.count(), sf.neval);
        if (vegas)
            std::fprintf(stderr, ", chi2 prob L %.3g T %.3g", sf.FL_chi2_prob, sf.FT_chi2_prob);
        std::fprintf(stderr, "\n");
        if (!vegas && sf.fail != 0)
            std::fprintf(stderr,
                         "Warning: %s: Cuhre did not reach epsrel = %g within maxeval = %lld "
                         "evaluations; the errors are its estimate at that point.\n",
                         calc, o.integration.cuhre.epsrel, o.integration.cuhre.maxeval);
    };

    run_one("OT",
            [&] { return dis::ot_structure_functions(o.kin, dipole, o.integration, o.sampling); });
    run_one("FE_4D", [&] {
        return dis::fe4d_structure_functions(o.kin, dipole, o.largeNc, o.integration, o.sampling);
    });
    run_one("FE_5D", [&] {
        return dis::fe5d_structure_functions(o.kin, dipole, o.largeNc, o.integration, o.sampling);
    });
}

}  // namespace

int main(int argc, char **argv) {
    try {
        const Options o = parse(argc, argv);
        // Add new interpolators here (see interpolation.hpp).
        if (o.interpolation == dis::BicubicSpline::name) run<dis::BicubicSpline>(o);
        else if (o.interpolation == dis::Bilinear::name) run<dis::Bilinear>(o);
        else throw std::invalid_argument("unknown interpolation " + o.interpolation);
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        usage(argv[0]);
        return 1;
    }
    return 0;
}
