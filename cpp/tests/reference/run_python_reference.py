"""Integrated reference values and timings from the Python implementation.

Makes the same calls as
src/small_x_physics/Cross_Sections/InclusiveDIS/withBK/test_bk_integration.py,
but under an `if __name__ == "__main__"` guard and with a wall-clock timer around
each stage.

The guard matters. On macOS multiprocessing starts workers with "spawn", which
re-imports the main script in every worker. test_bk_integration.py has no guard,
so each vegas worker re-runs the whole calculation: the table is printed many
times and the elapsed time measures that duplication, not the integration.

The number of vegas worker processes follows test_bk_integration.py: it is
taken from SLURM_CPUS_PER_TASK when set, otherwise from the core count. Use
SLURM_CPUS_PER_TASK=1 to time a single process.

    python3 cpp/tests/reference/run_python_reference.py --xB 1e-3 --mcpoints 100000
"""

import argparse
import json
import multiprocessing
import os
import time
import warnings

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--Q", type=float, default=np.sqrt(2.0))
    parser.add_argument("--xB", type=float, default=1e-3)
    parser.add_argument("--mf", type=float, default=0.14)
    parser.add_argument("--Zf", type=float, default=np.sqrt(2 / 3))
    parser.add_argument("--sigma0", type=float, default=2 * 2.57 * 18.81)
    parser.add_argument("--bkfile", type=str, default="data/mv.dat")
    parser.add_argument("--x0", type=float, default=0.01)
    parser.add_argument("--mcpoints", type=int, default=100000)
    parser.add_argument("--largeNc", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="gvar.ranseed for vegas")
    parser.add_argument("--only", choices=["OT", "FE_4D", "FE_5D"], nargs="*")
    parser.add_argument("--json", type=str, default=None, help="also write results here")
    parser.add_argument("--txt", type=str, default=None,
                        help="also write results as a text table, the format the C++ tests read")
    args = parser.parse_args()

    import gvar
    from small_x_physics.building_blocks.constants import alpha_em
    from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_OT_CS import OT_CrossSection_BK
    from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_FE_4D_evolve_in_xB import (
        FE_CrossSection_BK_4D,
    )
    from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_FE_5D_evolve_in_xP import (
        FE_CrossSection_BK_5D,
    )

    if args.seed is not None:
        gvar.ranseed(args.seed)
    warnings.simplefilter("ignore", RuntimeWarning)

    r_min, r_max = 1e-6, 10.0
    z_min = 1e-8
    z_max = 1.0 - z_min
    th_min, th_max = 0.0, 2 * np.pi
    M2_min = 4 * args.mf**2
    M2_max = args.Q**2 * (1 / args.xB - 1)

    common = dict(Q=args.Q, xB=args.xB, mf=args.mf, Zf=args.Zf, sigma0=args.sigma0,
                  Qs0=np.sqrt(0.104), gamma=1.0, ec=1.0, bkfile=args.bkfile,
                  x0=args.x0, mcpoints=args.mcpoints)
    prefactor = args.Q**2 / (4 * np.pi**2 * alpha_em)
    n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
    stages = args.only or ["OT", "FE_4D", "FE_5D"]

    results = []

    def record(calc, seconds, L, T):
        FL, FL_err = prefactor * L[0], prefactor * L[1]
        FT, FT_err = prefactor * T[0], prefactor * T[1]
        results.append(dict(
            calc=calc, Q2=args.Q**2, xB=args.xB, mf=args.mf, largeNc=int(args.largeNc),
            mcpoints=args.mcpoints, n_cores=n_cores, seconds=seconds,
            FL=FL, FL_err=FL_err, FT=FT, FT_err=FT_err,
            F2=FL + FT, F2_err=float(np.hypot(FL_err, FT_err)),
        ))

    if "OT" in stages:
        t0 = time.perf_counter()
        r = OT_CrossSection_BK(**common).BK_OT_cross_section(r_min, r_max, z_min, z_max)
        record("OT", time.perf_counter() - t0, r[0:2], r[2:4])

    if "FE_4D" in stages:
        t0 = time.perf_counter()
        res = {}
        for pol in ("L", "T"):
            cs = FE_CrossSection_BK_4D(polarization=pol, largeNc=args.largeNc, **common)
            res[pol] = cs.BK_FE_cross_section_4D(r_min, r_max, z_min, z_max, th_min, th_max)
        record("FE_4D", time.perf_counter() - t0, res["L"], res["T"])

    if "FE_5D" in stages:
        t0 = time.perf_counter()
        res = {}
        for pol in ("L", "T"):
            cs = FE_CrossSection_BK_5D(polarization=pol, largeNc=args.largeNc, **common)
            res[pol] = cs.BK_FE_cross_section_5D(
                r_min, r_max, z_min, z_max, th_min, th_max, M2_min, M2_max)
        record("FE_5D", time.perf_counter() - t0, res["L"], res["T"])

    keys = ["calc", "Q2", "xB", "mf", "largeNc", "mcpoints", "n_cores", "seconds",
            "FL", "FL_err", "FT", "FT_err", "F2", "F2_err"]
    print(", ".join(keys))
    for row in results:
        print(", ".join(repr(row[k]) for k in keys))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    if args.txt:
        write_txt(results, args.txt, seed=args.seed)


TXT_COLUMNS = ["calc", "Q2", "xB", "mf", "largeNc", "mcpoints", "n_cores", "seconds",
               "FL", "FL_err", "FT", "FT_err"]


def write_txt(results, path, seed=None):
    """Write results as '#'-commented, whitespace-separated columns."""
    with open(path, "w") as f:
        f.write("# Python vegas results, src/small_x_physics (run_python_reference.py)\n")
        f.write(f"# seed = {seed}\n")
        f.write("# " + " ".join(TXT_COLUMNS) + "\n")
        for row in results:
            f.write(" ".join(row["calc"] if k == "calc" else repr(float(row[k])) for k in TXT_COLUMNS))
            f.write("\n")


if __name__ == "__main__":
    main()
