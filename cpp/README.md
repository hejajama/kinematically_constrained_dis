# C++ implementation

A C++ port of the BK-evolved DIS structure function calculation of
`src/small_x_physics`. It computes the same three quantities as
`src/small_x_physics/Cross_Sections/InclusiveDIS/withBK/test_bk_integration.py`:

- `OT`: the optical theorem;
- `FE_4D`: finite-energy constrained, BK evolved in xB;
- `FE_5D`: finite-energy constrained, BK evolved in xP.

It takes the same command line options as that script and prints the same CSV table. Integration uses Cuhre (by
default) or Vegas from [Cuba 4.2.2](http://www.feynarts.de/cuba/), vendored in `external/`.

## Build

Needs CMake ≥ 3.16, a C++17 compiler and Boost (header-only Boost.Math, for the Bessel functions). Without Boost,
build with `-Duse_boost=OFF` to take the Bessel functions from GSL instead; they are 3–4× slower, which makes the FE
integrands about 1.8× slower.

```sh
cmake -S cpp -B cpp/build -Dunittest=ON
cmake --build cpp/build -j
```

The build is `Release` by default and uses `-march=native`; turn that off with `-Dnative=OFF`.

## Run

```sh
cpp/build/bin/dis_xs --bkfile data/mv.dat --xB 1e-3 --mcpoints 100000
```

`--help` lists all options. Besides the physics options of the Python script, it accepts:

- `--integrator cuhre|vegas`: the integration routine (see below);
- `--epsrel E`, `--maxeval N`: Cuhre's target relative error (default 1e-3) and evaluation limit (default 5e7);
- `--mcpoints N`, `--seed S`, `--nbatch`, `--iterations`, `--warmup`: Vegas settings; seed `0` selects Sobol
  quasi-random numbers;
- `--interpolation bicubic|bilinear`: the dipole interpolation;
- `--only OT|FE_4D|FE_5D`: compute only these rows; can be repeated;
- `--cores N`: number of worker processes; `0` means none;
- `--verbose V`: Cuba's verbosity, 0–3;
- `--variables python`: sample the same variables as the Python code (see below).

Timings, the number of evaluations and Cuhre warnings go to stderr. With Cuhre, the `mcpoints` column of the CSV holds
the number of integrand evaluations.

L and T are computed in one integration, as two components sharing each evaluation of the dipoles and the quadrupole.
The Python code runs a separate integration per polarization. Measured over 5 seeds, separate integrations give
10–15% smaller FE errors at equal points per polarization, but cost twice the time. At equal time, the combined run is
better.

By default Cuba evaluates the integrand in forked worker processes. Their number comes from `CUBACORES`, otherwise from
the number of idle cores.

### Integration routine

**Cuhre (default)** is deterministic adaptive cubature. It subdivides the integration region until the estimated error
is below `epsrel` times the result, or until `maxeval` evaluations, in which case it prints a warning. At xB = 1e-3 on
8 cores:

- **`epsrel` 1e-3:** OT takes milliseconds, FE_4D and FE_5D about 1 s each.
- **`epsrel` 1e-4:** OT still takes milliseconds, FE_4D and FE_5D about 7–10 s each.

The error estimate compares cubature rules of different degree. It is not statistical, so it cannot be checked by
changing the seed. Check it by repeating with a 10× smaller `epsrel`: at xB = 1e-3 it covered the difference with
room to spare. Cuhre keeps all subregions in memory, and beyond a few 10⁷ evaluations it slows down noticeably.

**Vegas** is adaptive Monte Carlo, as used by the Python code. As there, a warm-up of 10 iterations of `mcpoints/10`
points trains the grid and is discarded, then 20 iterations of `mcpoints` points give the result. Its errors are
statistical and can be checked over seeds.

Compared with Vegas at the same number of evaluations, Cuhre's errors are 20–25× smaller for FE. For OT, Cuhre gives
the integral to 1e-9 in 0.1 s, a precision Vegas cannot reach. Suave and Divonne from Cuba were also tested: Suave was
no better than Vegas for FE and was biased with small subdivisions, and Divonne was 25–450× slower.

### Integration variables

The C++ code computes the same integrals as the Python code over the same domains, but by default it samples them in
variables that the integrators handle better (`Sampling` in `src/cross_sections.hpp`). `--variables python` switches all of this
off, and single options can be switched off one at a time (see `--help`).

- **Relative coordinates for FE.** The phase-space kernel is sharply peaked at u⃗ = u⃗′, with a width of about
  1/√(M²_max z(1−z)) ≈ 0.05 GeV⁻¹ at xB = 1e-3. In the Python variables (u, u′, θ) the peak lies along a diagonal of the
  integration box, where Vegas cannot adapt to it. There the integrand's variance is infinite: estimates drift with
  the number of points, and the reported errors are too small. The C++ code integrates over u, r = |u⃗ − u⃗′| (log
  scale) and the angle of u⃗ − u⃗′ instead.
- **FE_5D integrated by parts in M²_qq.** In the original form, J0 oscillates over the whole M² range up to W², and the
  M² integral builds the peak above only by cancellation. Integrating by parts gives the FE_4D integrand with the
  dipole at xP = 1, plus a correction proportional to dT/dY (T the target amplitude). The correction vanishes
  wherever the dipole does not depend on Y: for a BK solution that is xP > x0, most of the M² range. The derivation is
  in the comment of `FE5DCorrectionIntegrand` in `src/integrands.hpp`. The correction gets a tenth of the points, since
  its variance is much smaller.
- **Symmetries.** z ↔ 1−z and angle ↔ −angle are exact symmetries, so z runs over [z_min, ½] and the angle over
  [0, π].
- **Dipole sizes on a log scale.**

At 1e6 points this makes the FE_4D error 1.8–1.9× smaller and the FE_5D error 12–16× smaller than with the Python
variables. Over repeated seeds, the reported errors match the actual scatter.

The numbers above are for Vegas. These variables are also what lets Cuhre work so well: in the Python variables, the
peaked and oscillating FE integrands are much harder for cubature too.

OT gains nothing from this with Vegas. Classic Vegas, which Cuba implements, cannot match the stratified sampling of
Python's vegas package for this smooth 2D integrand: at equal points, its error is 50–70× larger. Cuhre has no such
problem.

## Tests

```sh
cd cpp/build && ctest            # or: ./tests/dis_tests
```

ctest runs two entries:

- **`unit`** takes under a second. It compares every building block against values from the Python code at about 2000
  points each:
  - dipole spline;
  - wavefunctions;
  - quadrupole, with an analytic GBW dipole and with the BK dipole;
  - the three integrands, at finite and large Nc.

  It also tests exact properties: the spline reproduces polynomials and grid nodes, and the quadrupole has the right
  limits. It checks what the integration variables rely on: the symmetries, the relative coordinates, and the FE_5D
  integration by parts, against a 1D quadrature over M²_qq. Finally it checks the Cuhre and Vegas wrappers, including
  that the forked workers give the serial result.
- **`integration`** takes about 35 s on 8 cores. It runs the full integrations with Cuhre and with Vegas, with the
  default variables and with the Python ones, and compares them with the Python results. For Cuhre it also checks that
  the error estimate covers the change to a 10× smaller `epsrel`. OT is compared with a quadrature result. FE_4D and FE_5D
  must agree with the Python Monte Carlo results within 4 standard deviations of the combined error. For FE_5D the
  reference is the mean of three Python runs, with the error taken from their scatter.

### Reference data

`tests/reference/` holds the data the tests compare against and the scripts that produced it. Run the scripts from the
repository root, in the `kcdis` environment:

```sh
# pointwise values, deterministic (seeded); --quadrature adds the OT quadrature (~1 min)
python3 cpp/tests/reference/generate_reference.py --quadrature

# integrated Python results with timings
python3 cpp/tests/reference/run_python_reference.py --mcpoints 10000000 --seed 4 \
    --txt cpp/tests/reference/integrated_xB1e-3_mc1e7_8cores.txt
```

`run_python_reference.py` makes the same calls as `test_bk_integration.py`, but under an `if __name__ == "__main__"`
guard. On macOS, multiprocessing starts vegas workers with `spawn`, which re-imports the main script. Without the guard,
each worker re-runs the whole calculation.

## Code layout

| file | contents |
|---|---|
| `src/rcbk_table.*` | reads an rcbk solver output file: grids, N(r, Y), header |
| `src/interpolation.*` | interpolators of N(r, Y): `BicubicSpline` (as scipy's `RectBivariateSpline`), `Bilinear` |
| `src/dipole.hpp` | `BKDipole<Interpolator>` (clamping, clipping, S floor), analytic `GBWDipole` |
| `src/wavefunctions.hpp` | LO photon wavefunctions squared, OT and FE |
| `src/quadrupole.hpp` | Gaussian quadrupole from six dipoles, finite and large Nc |
| `src/integrands.hpp` | OT, FE 4D and FE 5D integrands, the FE 5D correction term of the integration by parts |
| `src/integration.*` | Cuba Cuhre and Vegas wrappers, axis mappings |
| `src/cross_sections.hpp` | structure functions from the integrands; integration variables (`Sampling`) |
| `src/special_functions.*` | Bessel functions (Boost.Math, or GSL with `-Duse_boost=OFF`) |
| `src/main.cpp` | command line program |

### Changing the dipole interpolation

The interpolator is a template parameter of `BKDipole`. To add one:

1. Write a class with the interface described at the top of `src/interpolation.hpp`:
   - a constructor from `RcbkTable`;
   - a `name`;
   - `at(Y)` returning a slice that evaluates N(r) at that rapidity.

   `BKDipole` handles everything outside the grid, so the class only interpolates inside it.
2. Add it to the dispatch in `main()` in `src/main.cpp`.

The quadrupole, the integrands and the cross sections only need `dipole.at(Y).S(r)`. A dipole that is not tabulated at
all can be used the same way, as `GBWDipole` is in the tests.

## Notes on the Cuba build

Two problems in the vendored Cuba sources are worked around in `external/cuba-4.2.2/CMakeLists.txt`. Without either fix,
the parallel workers either hang or return 0:

- `Fork.c` starts the worker processes inside an `assert()`. A `Release` build defines `NDEBUG`, which removes the
  `fork()` call, so Cuba is compiled with `-UNDEBUG`.
- `llVegas` (64-bit evaluation counts) needs `Fork.c` compiled with `LONGLONGINT` as well, so all of Cuba is.
