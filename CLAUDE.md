# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Micki is an object-oriented microkinetic modeling package in Python. Users build `Gas`/`Liquid`/`Adsorbate`/`Electron` species (thermochemistry from ASE `Atoms` + vibrational frequencies), combine them into `Reaction`s, add those to a `Model`, and integrate the resulting DAE system with SUNDIALS IDA through the official Python interface, sundials4py.

There is no build system, packaging (`setup.py`), or linter config. The package is used by putting the repo root on `PYTHONPATH` and running `import micki`.

## Dependencies / environment

- Python ≥3.12, `sundials4py` (≥7.9, beta; wheels bundle SUNDIALS, so no compiler or LAPACK is needed), `numpy` ≥2, `sympy`, `ase`.
- Older implementations that generated Fortran, compiled it with f2py and linked SUNDIALS + LAPACK are on branches `fortran-sundials7` (SUNDIALS 7, F2003 interface; last Fortran version, includes all bug fixes and the regression test) and `sundials4` (SUNDIALS 4.X FCMIX).

## Tests

```
python -m unittest discover -s tests -v                                   # all (~40 s)
python -m unittest discover -s tests -k test_difference_quotient_jacobian  # one test (~20 s)
```

`tests/test_wgs.py` is a regression test on a water-gas-shift model (`tests/wgs.py`, database `tests/data/wgs.json`): for 21 reaction conditions it solves a CSTR to steady state, then a PFR, and compares TOFs, CSTR steady states and PFR outlet states with `tests/data/wgs_reference.json` (rtol 1e-6), once with each Jacobian mode. The reference was generated with the original Fortran/SUNDIALS 4.X implementation; the current solver reproduces it to ~2e-10. After an intentional change in results, regenerate it with `python tests/wgs.py` and explain the change in the commit message.

## Architecture

**Species (`reactants.py`)** — `_Thermo` is the base class computing partition functions and H/S/G/E at temperature `T`, with caching via `update()`/`is_update_needed()`. Subclasses: `_Fluid` → `Gas`, `Liquid`; `Adsorbate` (occupies `sites`, may be a transition state); `Electron`. Each species with a label gets a sympy `Symbol` (`species.symbol`) used to build rate expressions. Species support `+` and `*` to form `_Reactants` collections, which is how reaction sides are written (e.g. `2 * H + O2`). Site vacancies are themselves `Adsorbate`-like species referenced via `sites`.

**Reactions (`model.py: Reaction`)** — computes `keq`, `kfor`, `krev` (possibly sympy expressions in coverages, for lateral interactions/lattice effects). The rate law is selected by `method`: `TST` (default when `ts` given), `EQUIL` (default otherwise), `DIEQUIL`, `STICK`, `ER`, `DIFF`, `DIFF_LIQ`. Per-parameter multipliers are in `reaction.scale` (used by `analysis.py` for sensitivity/degree-of-rate-control).

**Model (`model.py: Model`)** — the core pipeline lives in `set_initial_conditions(U0)`:
1. Reorders species (Liquid → Gas/Electron → Adsorbate); fixed species and the solvent are excluded from the ODE variables.
2. Computes vacancy concentrations from site balances (and `Lattice` site ratios if provided).
3. Builds symbolic rate expressions with sympy, substitutes fixed concentrations, and symbolically differentiates to get the Jacobian (`drdy`, `drdvac`).
4. Builds an `IDASolver` (`solver.py`): the rate, vacancy and derivative expressions become NumPy functions via `sympy.lambdify(..., cse=True)`, and IDA (dense `SUNLinSol_Dense`) is set up with the residual F = dypdr·r(y, vac(y)) − id·y′ as a Python callback.
5. `solve(t, ncp)` and `find_steady_state()` call into the `IDASolver` and convert arrays back to dicts keyed by species/reaction name.

Consequences to keep in mind:
- Changing T, Asite, z, or lattice on an initialized model re-runs `set_initial_conditions` (rebuilding the symbolic expressions and the solver).
- `reactor='PFR'` zeroes the mass-matrix entries of adsorbates (algebraic, pseudo-steady-state); `'CSTR'` treats all variables as differential.
- By default no Jacobian function is attached, so IDA uses its difference-quotient Jacobian (as the old Sundials 4.X/FCMIX version did). `Model(..., analytic_jac=True)` registers `IDASolver.jacobian` (dypdr·(∂r/∂y + ∂r/∂vac·∂vac/∂y) − c_j·M). `model._solver.residual(y, yp)` and `.jacobian(y, cj)` can be called directly, e.g. to check the Jacobian against finite differences.
- The residual returns a recoverable error when any y < −1e-10; negative vacancies are clipped to 0 in the rates, and negative y and vacancies are clipped (with an error flag) in the Jacobian. These rules come from the original Fortran template.
- `find_steady_state(epsilon=...)` stops when max |dy/dt| < epsilon over the differential variables, with dy/dt taken from IDA's y′. Do not re-evaluate dy/dt from the rate expressions for this: near-equilibrium steps with fluxes ~1e8 give ~1e-8–1e-7 rounding noise, which made the old check (and the Fortran versions) run to `maxiter` at random.
- If any rate expression (e.g. via `species.lateral`) refers to a species that is not in the model, `set_initial_conditions` raises `ValueError` naming the species and reactions; such symbols used to be silently set to 0.

**Supporting modules**
- `analysis.py: ModelAnalysis` — steady-state analysis: Campbell degree of rate control, thermodynamic rate control, apparent activation barrier, reaction orders (finite differences on `scale`/T/concentrations).
- `eref.py: EnergyReference` — solves a linear system for per-element reference energies from N structures containing N elements.
- `db.py` — round-trips species to/from an ASE database (`read_from_db`, `_Thermo.save_to_db`); rows store `freqs`, `thermo`, `sites`, `ts`, etc. in `row.data`.
- `lattice.py: Lattice` — site neighbor lists for configurational entropy.
- `io.py` — VASP output parsing; `masses.py` — atomic mass table.

## Conventions

- Python 3, 4-space indentation (a past commit converted tabs to spaces; don't reintroduce tabs).
- Units follow ASE (`ase.units`): energies in eV.
