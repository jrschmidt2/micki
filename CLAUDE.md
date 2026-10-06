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

`tests/test_wgs.py` is a regression test on a water-gas-shift model (`tests/wgs.py`, database `tests/data/wgs.json`): for 21 reaction conditions it solves a CSTR to steady state, then a PFR, and compares TOFs, CSTR steady states and PFR outlet states with `tests/data/wgs_reference.json` (rtol 1e-6), once with each Jacobian mode. Conditions follow Table 5 of Grabow et al., J. Phys. Chem. C 2008, 112, 4608 (the original `wgs_tof.py` had y(H2) of condition 13 and y(CO) of condition 17 mistyped). Apart from those two conditions, the reference agrees with the original Fortran/SUNDIALS 4.X results to ~1e-10. After an intentional change in results, regenerate it with `python tests/wgs.py` and explain the change in the commit message.

## Architecture

**Species (`reactants.py`)** — `_Thermo` is the base class computing partition functions and H/S/G/E at temperature `T`, with caching via `update()`/`is_update_needed()`. Subclasses: `_Fluid` → `Gas`, `Liquid`; `Adsorbate` (occupies `sites`, may be a transition state); `Electron`. Each species with a label gets a sympy `Symbol` (`species.symbol`) used to build rate expressions. Species support `+` and `*` to form `_Reactants` collections, which is how reaction sides are written (e.g. `2 * H + O2`). Site vacancies are themselves `Adsorbate`-like species referenced via `sites`.

**Reactions (`model.py: Reaction`)** — computes `keq`, `kfor`, `krev` (possibly sympy expressions in coverages, for lateral interactions/lattice effects). The rate law is selected by `method`: `TST` (default when `ts` given), `EQUIL` (default otherwise), `DIEQUIL`, `STICK`, `ER`, `DIFF`, `DIFF_LIQ`. Per-parameter multipliers are in `reaction.scale` (used by `analysis.py` for sensitivity/degree-of-rate-control).

**Model (`model.py: Model`)** — the core pipeline lives in `set_initial_conditions(U0)`:
1. Reorders species (Liquid → Gas/Electron → Adsorbate); fixed species and the solvent are excluded from the ODE variables.
2. Computes vacancy concentrations from site balances (and `Lattice` site ratios if provided).
3. Builds symbolic rate expressions with sympy. Fixed species (and the solvent) stay symbolic; their concentrations are passed to the solver at evaluation time instead of being substituted (substituting with `subs()` was slow).
4. Builds an `IDASolver` (`solver.py`): the rate and vacancy expressions become NumPy functions via `sympy.lambdify(..., cse=True)`, and IDA (dense `SUNLinSol_Dense`) is set up with the residual F = dypdr·r(y, vac(y), c_fixed) − id·y′ as a Python callback. There are no symbolic derivatives: the Jacobian is computed by complex-step differentiation of the same rate functions (all n perturbations in one vectorized call), exact to rounding. `Piecewise`, `Max`, `Min`, `Heaviside` work with it as is; `Abs` is mapped to a complex-step-safe version (`_LAMBDIFY_MODULES` in `solver.py`); `re`, `im`, `conjugate`, `arg` of variables are not supported. `tests/test_jacobian.py` checks these.
5. `solve(t, ncp)` and `find_steady_state()` call into the `IDASolver` and convert arrays back to dicts keyed by species/reaction name.

Consequences to keep in mind:
- Changing T, Asite, z, or lattice on an initialized model re-runs `set_initial_conditions` (rebuilding the symbolic expressions and the solver).
- `reactor='PFR'` zeroes the mass-matrix entries of adsorbates (algebraic, pseudo-steady-state); `'CSTR'` treats all variables as differential.
- By default no Jacobian function is attached, so IDA uses its difference-quotient Jacobian (as the old Sundials 4.X/FCMIX version did). `Model(..., analytic_jac=True)` registers the exact (complex-step) `IDASolver.jacobian` (dF/dy − c_j·M); Newton always uses it. `model._solver.residual(y, yp)` and `.jacobian(y, cj)` can be called directly, e.g. to check the Jacobian against finite differences.
- The residual returns a recoverable error when any y < −1e-10; negative vacancies are clipped to 0 in the rates, and negative y and vacancies are clipped (with an error flag) in the Jacobian. These rules come from the original Fortran template.
- `find_steady_state()` (default `method='hybrid'`) first solves dypdr·r(y) = 0 directly with `IDASolver.newton()`: Newton in ln(y) with the analytic Jacobian, line search on the residual scaled by gross flux, converged when the full step is < 1e-10 relative. It only converges from a nearby guess (e.g. the unperturbed steady state in sensitivity scans); otherwise it integrates in time and then polishes with Newton. `model.steady_state_method` records the path ('newton', 'integrate+newton', 'integrate'); t is `inf` when Newton alone was used. Models with conserved quantities (rank(dypdr) < n, e.g. no fixed species) always integrate. `method='integrate'` gives the integration-only behavior.
- The integration path stops when max |dy/dt| < epsilon, with dy/dt taken from IDA's y′. Do not re-evaluate dy/dt from the rate expressions for this: near-equilibrium steps with fluxes ~1e8 give ~1e-8–1e-7 rounding noise, which made the old check (and the Fortran versions) run to `maxiter` at random.
- `check_rates()` (run after every solve) evaluates coverage-dependent rate constants through cached `lambdify` functions; sympy `subs()` there used to dominate `find_steady_state`.
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
