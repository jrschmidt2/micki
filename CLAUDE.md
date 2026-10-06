# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Micki is an object-oriented microkinetic modeling package in Python. Users build `Gas`/`Liquid`/`Adsorbate`/`Electron` species (thermochemistry from ASE `Atoms` + vibrational frequencies), combine them into `Reaction`s, add those to a `Model`, and integrate the resulting DAE system with SUNDIALS IDA via a Fortran module that is generated and compiled at runtime.

There is no build system, packaging (`setup.py`), test suite, or linter config. The package is used by putting the repo root on `PYTHONPATH` and running `import micki`.

## Dependencies / environment

- Python 3 **with development headers** (`Python.h`; f2py builds a C extension at runtime), `ase`, `sympy`, and `numpy<2` (`numpy.f2py.compile` was removed in numpy 2.0; on Python ≥3.12 the distutils backend it relies on is also gone, so use Python ≤3.11)
- A Fortran compiler usable by `numpy.f2py`
- SUNDIALS ≥7.0 (tested with 7.9.0), built with `-DSUNDIALS_ENABLE_FORTRAN=ON -DSUNDIALS_ENABLE_LAPACK=ON -DSUNDIALS_INDEX_SIZE=32 -DCMAKE_POSITION_INDEPENDENT_CODE=ON`. The generated Fortran uses the Fortran 2003 interface (`fida_mod`, `fnvector_serial_mod`, ...); the `.mod` files must come from the same gfortran version that compiles micki's module. Set `MICKI_SUNDIALS_DIR` to the install prefix so f2py gets `-I$DIR/fortran -L$DIR/lib64`. Index size must be 32: the template passes `c_int32_t` lengths, and SUNDIALS' LAPACK solver passes its index type straight to LAPACK.
- LAPACK: defaults to MKL (`-lmkl_rt`); override with the `MICKI_LAPACK` env var (e.g. `MICKI_LAPACK="-lmkl_gf_lp64 -lmkl_sequential -lmkl_core"`). Use an LP64 (32-bit integer) LAPACK to match `SUNDIALS_INDEX_SIZE=32`. SUNDIALS and LAPACK library dirs must be on both `LIBRARY_PATH` (link) and `LD_LIBRARY_PATH` (import).
- Compiler/linker errors from f2py are mostly suppressed (`-w`, `--quiet`); a failed build surfaces only as `ModuleNotFoundError: No module named 'tmpXXXX'`. To see the real error, run `python -m numpy.f2py -c <pyf> solve_ida.f90 -l...` by hand on the `solve_ida.f90` left in the working directory.

There are no tests or examples in this repo. A water-gas-shift example (`wgs_tof.py` + `wgs.json`) is used as an end-to-end check; it computes TOFs for 21 conditions and compares them with reference values in the script.

See README.md for step-by-step SUNDIALS installation.

## Architecture

**Species (`reactants.py`)** — `_Thermo` is the base class computing partition functions and H/S/G/E at temperature `T`, with caching via `update()`/`is_update_needed()`. Subclasses: `_Fluid` → `Gas`, `Liquid`; `Adsorbate` (occupies `sites`, may be a transition state); `Electron`. Each species with a label gets a sympy `Symbol` (`species.symbol`) used to build rate expressions. Species support `+` and `*` to form `_Reactants` collections, which is how reaction sides are written (e.g. `2 * H + O2`). Site vacancies are themselves `Adsorbate`-like species referenced via `sites`.

**Reactions (`model.py: Reaction`)** — computes `keq`, `kfor`, `krev` (possibly sympy expressions in coverages, for lateral interactions/lattice effects). The rate law is selected by `method`: `TST` (default when `ts` given), `EQUIL` (default otherwise), `DIEQUIL`, `STICK`, `ER`, `DIFF`, `DIFF_LIQ`. Per-parameter multipliers are in `reaction.scale` (used by `analysis.py` for sensitivity/degree-of-rate-control).

**Model (`model.py: Model`)** — the core pipeline lives in `set_initial_conditions(U0)`:
1. Reorders species (Liquid → Gas/Electron → Adsorbate); fixed species and the solvent are excluded from the ODE variables.
2. Computes vacancy concentrations from site balances (and `Lattice` site ratios if provided).
3. Builds symbolic rate expressions with sympy, substitutes fixed concentrations, and symbolically differentiates to get the Jacobian (`drdy`, `drdvac`).
4. `setup_execs()` emits Fortran via `sym.fcode`, fills `f90_template`/`pyf_template` from `fortran.py`, compiles with `numpy.f2py.compile` (linking SUNDIALS + LAPACK) in a temp dir, imports the module by its random name, and then deletes the `.so`.
5. Calls the Fortran `initialize`; `solve(t, ncp)` and `find_steady_state()` call into the compiled module and convert arrays back to dicts keyed by species/reaction name.

Consequences to keep in mind:
- Changing T, Asite, z, or lattice on an initialized model re-runs `set_initial_conditions` (i.e. recompiles).
- `setup_execs()` writes `solve_ida.f90` into the **current working directory** as a debugging aid; inspect it when the generated Fortran fails to compile. f2py stderr is printed only on failure.
- `reactor='PFR'` zeroes the mass-matrix entries of adsorbates (algebraic, pseudo-steady-state); `'CSTR'` treats all variables as differential.
- SUNDIALS handles live in the `ida_state` Fortran module (not exposed through the `.pyf`); `micki_resfn` is the `bind(C)` residual callback wrapping `fidaresfun`. By default no Jacobian function is attached, so IDA uses its difference-quotient Jacobian (as the old Sundials 4.X/FCMIX version did). `Model(..., analytic_jac=True)` registers `micki_jacfn`, which wraps the symbolically derived `fidadjac`. `calc_res`/`calc_jac` are exposed on the compiled module (`model._solve_ida`) for checking the Jacobian against finite differences.
- If any rate expression (e.g. via `species.lateral`) refers to a species that is not in the model, `set_initial_conditions` raises `ValueError` naming the species and reactions; such symbols used to be silently set to 0.
- `CFLAGS="-w -std=c99"` is set as a gcc workaround.

**Supporting modules**
- `analysis.py: ModelAnalysis` — steady-state analysis: Campbell degree of rate control, thermodynamic rate control, apparent activation barrier, reaction orders (finite differences on `scale`/T/concentrations).
- `eref.py: EnergyReference` — solves a linear system for per-element reference energies from N structures containing N elements.
- `db.py` — round-trips species to/from an ASE database (`read_from_db`, `_Thermo.save_to_db`); rows store `freqs`, `thermo`, `sites`, `ts`, etc. in `row.data`.
- `lattice.py: Lattice` — site neighbor lists for configurational entropy.
- `io.py` — VASP output parsing; `masses.py` — atomic mass table.

## Conventions

- Python 3, 4-space indentation (a past commit converted tabs to spaces; don't reintroduce tabs).
- Units follow ASE (`ase.units`): energies in eV.
