"""Generate pointwise reference values for the C++ unit tests.

Evaluates every building block of the Python implementation (BK dipole spline,
photon wavefunctions, quadrupole, and the OT / FE-4D / FE-5D integrands) at a
fixed, seeded set of points and writes them as whitespace-separated text files
next to this script. The C++ tests read these files and compare value by value.

Run from the repository root with the kcdis environment active:

    python3 cpp/tests/reference/generate_reference.py

Every file starts with '#' comment lines: the parameters used and the column
names. Values are written with 17 significant digits so that doubles round-trip.
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np

from small_x_physics.building_blocks.constants import Nc, alpha_em, LambdaQCD
from small_x_physics.building_blocks.correlators.Dipoles.BK_dipole import BKDipole
from small_x_physics.building_blocks.correlators.Quadrupoles.QuadrupoleCorrelator import (
    QuadrupoleCorrelatorModel,
)
from small_x_physics.building_blocks.wavefunctions.FE_photon_wavefunctions.LO import (
    LO_FE_PhotonWF_squared,
)
from small_x_physics.building_blocks.wavefunctions.OT_photon_wavefunctions.LO import (
    LO_OT_PhotonWF_squared,
)
from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_OT_CS import OT_CrossSection_BK
from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_FE_4D_evolve_in_xB import (
    FE_CrossSection_BK_4D,
)
from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_FE_5D_evolve_in_xP import (
    FE_CrossSection_BK_5D,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
BKFILE = REPO_ROOT / "data" / "mv.dat"

# Defaults of test_bk_integration.py
PARAMS = dict(
    Q=np.sqrt(2.0),
    xB=1e-3,
    mf=0.14,
    Zf=np.sqrt(2 / 3),
    sigma0=2 * 2.57 * 18.81,
    Qs0=np.sqrt(0.104),
    gamma=1.0,
    ec=1.0,
    bkfile=str(BKFILE),
    x0=0.01,
    mcpoints=1000,
)

# Integration limits of test_bk_integration.py
R_MIN, R_MAX = 1e-6, 10.0
Z_MIN = 1e-8
Z_MAX = 1.0 - Z_MIN


def write(name, columns, data, comments=()):
    data = np.column_stack(data)
    path = HERE / name
    with open(path, "w") as f:
        for line in comments:
            f.write(f"# {line}\n")
        f.write("# " + " ".join(columns) + "\n")
        np.savetxt(f, data, fmt="%.17g")
    print(f"wrote {path.relative_to(REPO_ROOT)}: {data.shape[0]} rows")


def param_comments(extra=()):
    lines = [f"{k} = {float(v)!r}" for k, v in PARAMS.items() if k not in ("bkfile", "mcpoints")]
    lines.append("bkfile = data/mv.dat")
    lines.append(f"Nc = {Nc}, alpha_em = {alpha_em!r}, LambdaQCD = {LambdaQCD!r}")
    return lines + list(extra)


def sample_dipole_sizes(rng, n, lo=R_MIN, hi=R_MAX):
    """Log-uniform sizes, which is where the integrands have their structure."""
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def sample_z(rng, n):
    """Mostly uniform z, plus a tail pushed towards the endpoints."""
    z = rng.uniform(Z_MIN, Z_MAX, n)
    k = n // 5
    z[:k] = np.exp(rng.uniform(np.log(Z_MIN), np.log(0.5), k))
    z[k:2 * k] = 1.0 - np.exp(rng.uniform(np.log(Z_MIN), np.log(0.5), k))
    return z


# ----------------------------------------------------------------------------
# BK dipole
# ----------------------------------------------------------------------------

def dipole(rng, n):
    d = BKDipole(BKFILE, warn_on_clamp=False)

    r = np.exp(rng.uniform(np.log(d.r_min), np.log(d.r_max), n))
    Y = rng.uniform(d.Y_min, d.Y_max, n)

    # Grid nodes and cell midpoints: the interpolant must hit the data exactly
    # at nodes, and midpoints are where any coefficient error is largest.
    iY = rng.integers(0, len(d.Y_grid), 50)
    ir = rng.integers(0, len(d.r_grid), 50)
    r = np.concatenate([r, d.r_grid[ir], 0.5 * (d.r_grid[ir[:-1]] + d.r_grid[ir[:-1] + 1])])
    Y = np.concatenate([Y, d.Y_grid[iY], 0.5 * (d.Y_grid[iY[:-1]] + d.Y_grid[iY[:-1] + 1])])

    # Out-of-range queries, which must clamp to the boundary.
    r_out = np.array([1e-8, 5e-7, 150.0, 1e3, 1e-3, 1e-3, 1.0, 1.0, 1e-8, 1e3])
    Y_out = np.array([2.0, 2.0, 2.0, 2.0, -1.0, 17.0, -0.5, 20.0, -1.0, 20.0])
    r = np.concatenate([r, r_out])
    Y = np.concatenate([Y, Y_out])

    N_table = d.N(r, Y)  # per-cell polynomial table, what the integrands use
    rc = np.clip(r, d.r_min, d.r_max)
    Yc = np.clip(Y, d.Y_min, d.Y_max)
    N_fitpack = np.clip(d.spline(Yc, rc, grid=False), 0.0, 1.0)
    S = d.S_r(r, Y)

    write(
        "dipole_N.txt",
        ["r", "Y", "N_table", "N_fitpack", "S"],
        [r, Y, N_table, N_fitpack, S],
        [
            "BKDipole(data/mv.dat) defaults: clamp_N=True, S_floor=1e-14, out of range clamps",
            f"grid: r in [{d.r_min!r}, {d.r_max!r}] ({len(d.r_grid)} pts), "
            f"Y in [{d.Y_min!r}, {d.Y_max!r}] ({len(d.Y_grid)} pts)",
            f"max |N_table - N_fitpack| = {np.max(np.abs(N_table - N_fitpack)):.3e}",
        ],
    )


# ----------------------------------------------------------------------------
# Photon wavefunctions
# ----------------------------------------------------------------------------

def wavefunctions(rng, n):
    Q, mf, Zf = PARAMS["Q"], PARAMS["mf"], PARAMS["Zf"]

    ot = LO_OT_PhotonWF_squared(Q, mf, Zf, Nc=Nc, alpha_em=alpha_em)
    r = sample_dipole_sizes(rng, n)
    z = sample_z(rng, n)
    write(
        "wf_OT.txt",
        ["r", "z", "psi_L_sq", "psi_T_sq"],
        [r, z, ot.psi_L_squared(Q, r, z), ot.psi_T_squared(Q, r, z)],
        param_comments(),
    )

    fe = LO_FE_PhotonWF_squared(mf, Zf, Nc=Nc, alpha_em=alpha_em)
    u = sample_dipole_sizes(rng, n)
    up = sample_dipole_sizes(rng, n)
    z = sample_z(rng, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    write(
        "wf_FE.txt",
        ["u", "up", "z", "theta", "psi_L_sq", "psi_T_sq"],
        [u, up, z, theta,
         fe.psi_L_squared(Q, u, up, z, theta), fe.psi_T_squared(Q, u, up, z, theta)],
        param_comments(),
    )


# ----------------------------------------------------------------------------
# Quadrupole
# ----------------------------------------------------------------------------

def gbw_S(x, y, Qs2=1.0):
    """Analytic GBW dipole, so the quadrupole is also tested independently of the spline."""
    diff = np.asarray(x) - np.asarray(y)
    r2 = diff[..., 0] ** 2 + diff[..., 1] ** 2
    return np.exp(-r2 * Qs2 / 4)


def quadrupole_points(rng, n):
    u = sample_dipole_sizes(rng, n)
    up = sample_dipole_sizes(rng, n)
    z = sample_z(rng, n)
    theta = rng.uniform(0, 2 * np.pi, n)

    # Special configurations:
    #  - u = up, theta = 0: x1 = x1', x2 = x2', so the quadrupole is S(u)^2-like
    #    and F1 = F2 = F3 = 0 up to round-off (the Delta = 0 branch).
    #  - tiny sizes: every dipole is 1, the transparent limit.
    #  - theta = pi, z = 1/2: a symmetric configuration.
    #  - very large sizes: dipoles hit S_floor, logs are ~ -32.
    special_u = np.array([0.5, 2.0, 1e-6, 1e-5, 1.0, 3.0, 9.0, 10.0])
    special_up = np.array([0.5, 2.0, 1e-6, 2e-5, 1.0, 1.0, 9.5, 10.0])
    special_z = np.array([0.3, 0.5, 0.5, 0.2, 0.5, 0.5, 0.4, 0.5])
    special_th = np.array([0.0, 0.0, 1.0, 2.0, np.pi, np.pi, 0.3, np.pi])
    return (
        np.concatenate([u, special_u]),
        np.concatenate([up, special_up]),
        np.concatenate([z, special_z]),
        np.concatenate([theta, special_th]),
    )


def quadrupole(rng, n):
    u, up, z, theta = quadrupole_points(rng, n)

    # GBW dipole: tests the Gaussian quadrupole algebra alone.
    gbw = QuadrupoleCorrelatorModel(Nc=Nc, LambdaQCD=LambdaQCD, dipole_model=gbw_S)
    Su, Sup, S4_F = gbw.polar_correlators(u, up, z, theta, largeNc=False)
    _, _, S4_L = gbw.polar_correlators(u, up, z, theta, largeNc=True)
    write(
        "quadrupole_gbw.txt",
        ["u", "up", "z", "theta", "S_u", "S_up", "S4_finiteNc", "S4_largeNc"],
        [u, up, z, theta, Su, Sup, S4_F, S4_L],
        [f"dipole S(r) = exp(-r^2 Qs^2 / 4), Qs^2 = 1, Nc = {Nc}, log floor 1e-14"],
    )

    # BK dipole at a per-point rapidity, as in the 5D integrand.
    d = BKDipole(BKFILE, warn_on_clamp=False)
    Y = rng.uniform(0.0, 16.0, len(u))
    bk = QuadrupoleCorrelatorModel(Nc=Nc, LambdaQCD=LambdaQCD, dipole_model=d.BK_evolved_MV_model_S2_Y)
    Su, Sup, S4_F = bk.polar_correlators(u, up, z, theta, dipole_args={"Y": Y}, largeNc=False)
    _, _, S4_L = bk.polar_correlators(u, up, z, theta, dipole_args={"Y": Y}, largeNc=True)
    write(
        "quadrupole_bk.txt",
        ["u", "up", "z", "theta", "Y", "S_u", "S_up", "S4_finiteNc", "S4_largeNc"],
        [u, up, z, theta, Y, Su, Sup, S4_F, S4_L],
        ["BKDipole(data/mv.dat), per-point Y", f"Nc = {Nc}, log floor 1e-14"],
    )


# ----------------------------------------------------------------------------
# Integrands
# ----------------------------------------------------------------------------

def integrand_ot(rng, n):
    cs = OT_CrossSection_BK(**PARAMS)
    r = sample_dipole_sizes(rng, n)
    z = sample_z(rng, n)
    write(
        "integrand_OT.txt",
        ["r", "z", "f_L", "f_T"],
        [r, z, cs.BK_integrand(r, z, "L"), cs.BK_integrand(r, z, "T")],
        param_comments([f"Y = log(x0/xB) = {cs.Y!r}"]),
    )


def integrand_fe4d(rng, n):
    u = sample_dipole_sizes(rng, n)
    up = sample_dipole_sizes(rng, n)
    z = sample_z(rng, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    x = np.array([u, up, z, theta])

    cols, vals = ["u", "up", "z", "theta"], [u, up, z, theta]
    for largeNc in (False, True):
        for pol in ("L", "T"):
            cs = FE_CrossSection_BK_4D(polarization=pol, largeNc=largeNc, **PARAMS)
            cols.append(f"f_{pol}_{'largeNc' if largeNc else 'finiteNc'}")
            vals.append(cs(x))
    write("integrand_FE4D.txt", cols, vals, param_comments([f"Y = log(x0/xB) = {cs.Y!r}"]))


def integrand_fe5d(rng, n):
    Q, xB, mf = PARAMS["Q"], PARAMS["xB"], PARAMS["mf"]
    M2_min = 4 * mf**2
    M2_max = Q**2 * (1 / xB - 1)

    u = sample_dipole_sizes(rng, n)
    up = sample_dipole_sizes(rng, n)
    z = sample_z(rng, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    # Log-uniform in M^2 so that both the clamped Y = 0 region (xP > x0) and the
    # kinematically forbidden region (M^2 < mf^2 / (z(1-z))) are well populated.
    M2 = np.exp(rng.uniform(np.log(M2_min), np.log(M2_max), n))
    x = np.array([u, up, z, theta, M2])

    cols, vals = ["u", "up", "z", "theta", "Mqq_sq"], [u, up, z, theta, M2]
    for largeNc in (False, True):
        for pol in ("L", "T"):
            cs = FE_CrossSection_BK_5D(polarization=pol, largeNc=largeNc, **PARAMS)
            cols.append(f"f_{pol}_{'largeNc' if largeNc else 'finiteNc'}")
            vals.append(cs(x))
    write(
        "integrand_FE5D.txt",
        cols,
        vals,
        param_comments([f"Mqq_sq in [{M2_min!r}, {M2_max!r}]"]),
    )


def ot_quadrature():
    """OT structure functions by nested adaptive quadrature, not Monte Carlo.

    The OT integrand is two-dimensional and smooth, so scipy.integrate.quad
    reaches ~1e-10 relative accuracy. That is a much sharper target for the C++
    integration than a vegas result with its own statistical error. Takes about
    a minute. The integrand is symmetric under z -> 1 - z, so z runs over
    [z_min, 1/2] and the result is doubled; r is integrated in log(r).
    """
    from scipy.integrate import quad

    cs = OT_CrossSection_BK(**PARAMS)
    prefactor = float(PARAMS["Q"]) ** 2 / (4 * np.pi**2 * alpha_em)

    rows = []
    for pol in ("L", "T"):
        def inner(z):
            def f(log_r):
                r = np.exp(log_r)
                return r * cs.BK_integrand(np.array([r]), np.array([z]), pol)[0]
            return quad(f, np.log(R_MIN), np.log(R_MAX), epsabs=0, epsrel=1e-11, limit=400)[0]

        value, err = quad(inner, Z_MIN, 0.5, epsabs=0, epsrel=1e-10, limit=400, points=[1e-4, 1e-2])
        rows.append((2 * value * prefactor, 2 * err * prefactor))

    write(
        "integrated_OT_quadrature.txt",
        ["FL", "FL_quad_err", "FT", "FT_quad_err"],
        [np.array([rows[0][0]]), np.array([rows[0][1]]), np.array([rows[1][0]]), np.array([rows[1][1]])],
        param_comments([
            f"r in [{R_MIN}, {R_MAX}], z in [{Z_MIN}, 1 - {Z_MIN}]",
            "F_{L,T} = Q^2 / (4 pi^2 alpha_em) sigma_{L,T}, nested scipy quad",
        ]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--n", type=int, default=2000, help="random points per file")
    parser.add_argument("--quadrature", action="store_true",
                        help="also compute the OT structure functions by quadrature (~1 min)")
    args = parser.parse_args()

    # Out-of-range dipole queries are intentional here.
    warnings.simplefilter("ignore", RuntimeWarning)

    rng = np.random.default_rng(args.seed)
    dipole(rng, args.n)
    wavefunctions(rng, args.n)
    quadrupole(rng, args.n)
    integrand_ot(rng, args.n)
    integrand_fe4d(rng, args.n)
    integrand_fe5d(rng, args.n)
    if args.quadrature:
        ot_quadrature()


if __name__ == "__main__":
    main()
