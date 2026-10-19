# C++ implementation

A C++ port of the BK-evolved DIS structure function calculation of
`src/small_x_physics`. It computes the same three quantities as
`src/small_x_physics/Cross_Sections/InclusiveDIS/withBK/test_bk_integration.py`:

- `OT`: the optical theorem;
- `FE_4D`: finite-energy constrained, BK evolved in xB;
- `FE_5D`: finite-energy constrained, BK evolved in xP.

It takes the same command line options as that script and prints the same CSV table. Integration uses Vegas from
[Cuba 4.2.2](http://www.feynarts.de/cuba/), vendored in `external/`.

## Build

Needs CMake ≥ 3.16, a C++17 compiler and GSL.

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

- `--interpolation bicubic|bilinear`: the dipole interpolation;
- `--only OT|FE_4D|FE_5D`: compute only these rows; can be repeated;
- `--seed S`: random seed; `0` selects Sobol quasi-random numbers;
- `--cores N`: number of worker processes; `0` means none;
- `--nbatch`, `--iterations`, `--warmup`, `--verbose`: Vegas settings.

Timings go to stderr.

L and T come out of one Vegas run, as two components sharing each evaluation of the dipoles and the quadrupole. The
Python code runs a separate integration per polarization. As in the Python code, a warm-up of 10 iterations of
`mcpoints/10` points trains the Vegas grid and is discarded, then 20 iterations of `mcpoints` points give the result.

By default Cuba evaluates the integrand in forked worker processes. Their number comes from `CUBACORES`, otherwise from
the number of idle cores.

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
  limits. Finally it checks the Vegas wrapper, including that the forked workers give the serial result.
- **`integration`** takes about 15 s on 8 cores. It runs the full integrations and compares them with the Python
  results. OT is compared with a quadrature result; FE_4D and FE_5D are required to agree with the Python Monte Carlo
  result within 4 standard deviations of the combined error.

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
| `src/integrands.hpp` | OT, FE 4D and FE 5D integrands |
| `src/vegas.*` | Cuba Vegas wrapper |
| `src/cross_sections.hpp` | structure functions from the integrands |
| `src/special_functions.*` | Bessel functions (GSL) |
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
