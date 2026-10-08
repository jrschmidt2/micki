# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Micki is an object-oriented microkinetic modeling package in Python. Users build `Gas`/`Liquid`/`Adsorbate`/`Electron` species (thermochemistry from ASE `Atoms` + vibrational frequencies), combine them into `Reaction`s, add those to a `Model`, and integrate the resulting DAE system with SUNDIALS IDA through the official Python interface, sundials4py.

Packaged with `pyproject.toml` (setuptools): `pip install -e .` in a Python ≥3.12 environment, or `conda env create -f environment.yml` (conda-forge Python/numpy/sympy/ase; sundials4py and micki via pip, since neither is on conda-forge). Tested on Python 3.12 and 3.14. Published on PyPI as `micki` (see Releasing). `sundials4py` is pinned `<8` because its API is still beta. There is no linter config; `python -m pyflakes micki tests` is clean.

## Dependencies / environment

- Python ≥3.12, `sundials4py` (≥7.9, beta; wheels bundle SUNDIALS, so no compiler or LAPACK is needed), `numpy` ≥2, `sympy`, `ase`.
- Older implementations that generated Fortran, compiled it with f2py and linked SUNDIALS + LAPACK are on branches `fortran-sundials7` (SUNDIALS 7, F2003 interface; last Fortran version, includes all bug fixes and the regression test) and `sundials4` (SUNDIALS 4.X FCMIX).

## Tests

```
python -m unittest discover -s tests -v                                   # all (~15 s)
python -m unittest discover -s tests -k test_difference_quotient_jacobian  # one test
```

`tests/test_wgs.py` is a regression test on a water-gas-shift model (`tests/wgs.py`, database `tests/data/wgs.json`): for 21 reaction conditions it solves a CSTR to steady state, then a PFR, and compares TOFs, CSTR steady states and PFR outlet states with `tests/data/wgs_reference.json` (rtol 1e-6), once with each Jacobian mode. Conditions follow Table 5 of Grabow et al., J. Phys. Chem. C 2008, 112, 4608 (the original `wgs_tof.py` had y(H2) of condition 13 and y(CO) of condition 17 mistyped). Reference history: it reproduced the original Fortran/SUNDIALS 4.X results to ~1e-10, then conditions 13/17 were corrected, then the fluid pV = kT fix raised all TOFs ×1.33–1.45, then the `symm` sign fix (it used to add kB ln symm) together with WGS data corrections (symm = 2 kept only on the symmetric o-h-oh TS; co-oh TS given two sites like the other TSs) raised them ×1.3–3.2 (RMS log-error vs experiment 0.133 → 0.348 → 1.049), then the OH and COOH `dE` shifts were refitted to the experimental TOFs (least squares in ln TOF, as in Hermes et al. 2019): OH −0.217 → −0.228 eV, COOH −0.052 → +0.160 eV, RMS 0.131. After an intentional change in results, regenerate it with `python tests/wgs.py` and explain the change in the commit message.

## Releasing

- To release: bump `version` in `pyproject.toml`, commit and push, then publish a GitHub release tagged `v<version>` (e.g. `v2.0.1`) in the GitHub web UI. `.github/workflows/publish.yml` then checks that the tag matches the version, installs the package and runs the tests, builds and `twine check`s the sdist and wheel, and uploads them with PyPI trusted publishing (no API token).
- The PyPI trusted publisher for `micki` is: owner `jrschmidt2`, repository `micki`, workflow `publish.yml`, environment `pypi`. Renaming the workflow file or the environment breaks publishing until the publisher on PyPI matches again (publishers cannot be edited on PyPI: remove and re-add).
- If the publish job fails with `invalid-publisher`, its error annotation lists the token claims GitHub sent (repository, `workflow_ref`, environment); compare them with the PyPI publisher. After fixing the PyPI side, use "Re-run failed jobs" on the run; no new tag or release is needed. (This happened for v2.0.0: the publisher said `publish.yaml`.)
- PyPI versions are immutable: a version that was uploaded can never be replaced or reused (a failed upload does not use it up). The PyPI project page shows the README packaged at the release tag, so README changes appear there only with the next release.
- "Run workflow" (`workflow_dispatch`) in the Actions tab runs only the build and tests (tag check and publish job are skipped); use it to test changes to the workflow. The `download-artifact` step runs only in the publish job.
- The actions are on their Node 24 major versions (`checkout` v7, `setup-python` v7, `upload-artifact` v7, `download-artifact` v8); keep upload/download-artifact versions compatible when bumping. CI uses Python 3.12, the minimum in `requires-python`.

## Architecture

**Species (`reactants.py`)** — `_Thermo` is the base class computing partition functions and H/S/G/E at temperature `T`, with caching via `update()`/`is_update_needed()`. Subclasses: `_Fluid` → `Gas`, `Liquid`; `Adsorbate` (occupies `sites`, may be a transition state); `Electron`. Each species with a label gets a sympy `Symbol` (`species.symbol`) used to build rate expressions. Species support `+` and `*` to form `_Reactants` collections, which is how reaction sides are written (e.g. `2 * H + O2`). Site vacancies are themselves `Adsorbate`-like species referenced via `sites`.

Free energies follow Hermes' dissertation (UW–Madison 2018): fluids use G = E + kT − TS per molecule, i.e. the chemical potential −kT ln(q/N) at the reference concentration (eqs. 2.17, 2.23; `_Fluid._calc_q`); adsorbates and transition states use the Helmholtz energy −kT ln(σ q), where σ is the number of distinguishable orientations (eqs. 3.8–3.16, 6.1; Hermes et al., J. Chem. Phys. 151, 014112 (2019), eqs. 6–7): the `Lattice` counts the orientations of multi-site species (6 for a two-site species on a hexagonal lattice) and the symmetry number `symm` divides them (2 for an end-to-end symmetric species), i.e. S_conf = kB ln(N_lattice/symm). Single-site species have no orientations counted, so `symm` > 1 there only lowers q and warns. `tests/test_model.py::test_adsorption_equilibrium_constant` checks K = Q_ads/Q_gas against ASE.

**Reactions (`model.py: Reaction`)** — computes `keq`, `kfor`, `krev` (possibly sympy expressions in coverages, for lateral interactions/lattice effects). The rate law is selected by `method`: `TST` (default when `ts` given), `EQUIL` (default otherwise), `DIEQUIL`, `STICK`, `ER`, `DIFF`, `DIFF_LIQ`. Per-parameter multipliers are in `reaction.scale` (used by `analysis.py` for sensitivity/degree-of-rate-control). A TS's free energy follows the reactants' and products' coverage terms (lateral + dE) as (1 − α)·Σ_reactants + α·Σ_products, with α computed self-consistently from the forward and reverse barriers (thesis eqs. 3.58–3.60; it depends on the gas reference state for reactions with fluids); `alpha=w` fixes it, `explicit_ts=True` uses only the TS's own energy and `ts.lateral`. `clip=None` raises for negative zero-coverage barriers computed without dE/lateral terms; `'zero_coverage'` (formerly `dground`, which now raises) decides the clip once at θ = 0; `'coverage'` (TST, EQUIL, STICK) uses Max(ΔG‡, ΔG, 0) at the current coverages, CatMap's convention (EQUIL + clip = CatMap barrierless, STICK + clip = CatMap non-activated). `Gas(..., pref=1)` references the gas free energy to 1 bar instead of `rhoref` (M); rates are reference-independent, but ΔG, the computed α and clipped barriers are not; `micki.utils.bar_to_molar` converts pressures.

**Model (`model.py: Model`)** — the core pipeline lives in `set_initial_conditions(U0)`:
1. Reorders species (Liquid → Gas/Electron → Adsorbate); fixed species and the solvent are excluded from the ODE variables.
2. Computes vacancy concentrations from site balances (and `Lattice` site ratios if provided).
3. Builds symbolic rate expressions with sympy. Fixed species (and the solvent) stay symbolic; their concentrations are passed to the solver at evaluation time instead of being substituted (substituting with `subs()` was slow).
4. Builds an `IDASolver` (`solver.py`): the rate and vacancy expressions become NumPy functions via `sympy.lambdify(..., cse=True)`, and IDA (dense `SUNLinSol_Dense`) is set up with the residual F = dypdr·r(y, vac(y), c_fixed) − id·y′ as a Python callback. There are no symbolic derivatives: the Jacobian is computed by complex-step differentiation of the same rate functions (all n perturbations in one vectorized call), exact to rounding. `Piecewise`, `Max`, `Min`, `Heaviside` work with it as is; `Abs` is mapped to a complex-step-safe version (`_LAMBDIFY_MODULES` in `solver.py`); `re`, `im`, `conjugate`, `arg` of variables are not supported. `tests/test_jacobian.py` checks these. Real-valued evaluations (residual, rates, vacancies; ~70k residual calls per WGS run) use separate `lambdify(..., 'math')` versions on Python floats, ~4x faster than numpy on scalars; a call that raises (overflow, domain error) is repeated with the numpy version, and the math path is disabled per model if it disagrees with numpy at the initial state (`_check_fast_path`). Benchmark performance changes serially, never with other jobs running; per-call overhead in the residual dominates.
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
- `conventions.py` — `set_conventions('micki'|'catmap')`, context manager `conventions(...)` (a `ContextVar`). Species and reactions record `.conventions` at construction (copies keep it); under 'catmap' unset defaults become `Gas` pref=1, ASE standard atomic weights for all species, `Reaction` clip='coverage' (TST/EQUIL/STICK only), alpha=0.5 (unless explicit_ts), and `Reaction.ts_follows_dE = False` (the TS follows only lateral terms, so dE acts like a CatMap energy change, also in TRC). `DEFAULT` sentinel: explicit `clip=None`/`alpha=None` mean no clip / computed alpha. `Model.add_reactions` raises on mixed conventions; a lattice warns. db rows with rhoref 1 (or a pref) load with rhoref=None so the flag's default applies.
- `lateral.py: first_order` — writes CatMap's first-order lateral interactions (Σ_j F(θ_tot)·ε_ij·θ_j, response 'linear', 'piecewise_linear', 'smooth_piecewise_linear') into `species.lateral`; TS rows go with `Reaction(..., explicit_ts=True)`. `Piecewise` branches divide by `Max(θ, x0)` so numpy's evaluation of untaken branches stays finite.
- `io.py` — VASP output parsing; `masses.py` — atomic mass table.

`tests/test_catmap.py` checks the CatMap-convention options (`pref`, `clip='coverage'`, `alpha`, `explicit_ts`, `micki.lateral.first_order`) on a WGS variant (`tests/catmap_wgs.py`: 2 variants × 4 conditions) against CatMap 0.3.1's own steady state (`tests/data/catmap_wgs_reference.json`, agreement ~3e-14, test rtol 1e-10); it also checks that `tests/data/catmap_wgs_input.json` matches the current model. To regenerate after changing that model: `python tests/catmap_wgs.py` (micki env), then `python tests/catmap_reference.py` in a CatMap env (CatMap is not on PyPI and needs numpy < 1.24, e.g. Python 3.11; the script patches CatMap's CODATA-2010 kB/h to ASE's and integrates CatMap's d(theta)/dt with SciPy Radau, because CatMap's Newton solvers fail on this model).

`CatMapModelsTest` (same file) checks two made-up models in `tests/catmap_models.py` (NH3 synthesis with non-activated H2 and linear interactions; C2H4 hydrogenation with two site types, cross-site interactions, smooth piecewise-linear response; 3 conditions each) against CatMap solving them natively with its own thermochemistry (`tests/data/catmap_models_reference.json`): free energies 1e-12 eV, rate constants 1e-11 at CatMap's coverages (CatMap's non-activated prefactor uses CODATA-2010 amu and eV, a 5.3e-8 effect the test divides out), coverages 1e-9, net rates 1e-6 for steps with flux/net < 1e7. Only the adsorbate interaction matrix is taken from CatMap. Regenerate with `tests/catmap_models_reference.py <model> <index>` per condition in the CatMap env (1–6 min each; run via SLURM), then `--merge <dir>`.

Other tests: `tests/test_model.py` (Model, Reaction, species, Lattice, vdW radius), `tests/test_analysis.py` (ModelAnalysis against independent finite differences on the WGS CSTR), `tests/test_jacobian.py` (complex-step Jacobian with non-analytic functions, fast math path and its numpy fallback).

## Conventions

- Raise exceptions (`ValueError`/`TypeError`) for invalid input and use `warnings.warn` for warnings; no `assert` for validation, no `print`.
- Python 3, 4-space indentation (a past commit converted tabs to spaces; don't reintroduce tabs).
- Units follow ASE (`ase.units`): energies in eV.
