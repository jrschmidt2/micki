# CatMap conventions

[CatMap](https://github.com/SUNCAT-Center/catmap) is another widely used
microkinetic modeling code. Its defaults differ from micki's in reference
states, barrier handling and transition-state interactions. micki can switch
to CatMap's conventions, either all at once or option by option. That makes it
possible to reproduce a CatMap model in micki and to compare the two codes
rigorously.

## The conventions flag

```python
import micki

micki.set_conventions('catmap')          # for the rest of the program

with micki.conventions('catmap'):        # or only for objects built here
    sp = read_from_db('species.json')
    rxns = build_reactions(sp)
```

The setting applies when species and reactions are **constructed**. Each object
records it (`.conventions`), and copies keep it. Under `'catmap'`, arguments
that are not given default to CatMap's conventions:

| | micki (default) | CatMap conventions |
|---|---|---|
| Gas reference state | 1 M (`rhoref=1`) | 1 bar (`pref=1`) |
| Atomic masses | most abundant isotopes (`micki.masses`) | standard atomic weights (`ase.data.atomic_masses`) |
| Barriers | error if negative at zero coverage | `clip='coverage'`: Max(ΔG‡, ΔG, 0) at the current coverages (TST, EQUIL, STICK) |
| Steps without a transition state | `EQUIL` (barrier max(0, ΔG) at zero coverage) | `EQUIL` + clip: barrierless, max(0, ΔG) at the current coverages |
| Transition-state interactions | α computed self-consistently (BEP-like) | `alpha=0.5` (CatMap's `intermediate_state`) |
| Transition states follow `dE` shifts | yes (through α) | no (only lateral interactions) |
| Lattice | optional | warns (CatMap has no configurational entropy) |

**What that means in practice:**
- **Explicit arguments win.** `Reaction(..., clip=None)` gives no clipping,
  `alpha=None` a computed α, and `explicit_ts=True` the TS's own interaction
  energy.
- **No mixing.** Reactions and species built under different conventions
  cannot be combined in one `Model`.
- **Stored reference states load as stored.** Databases written by micki ≥ 2.1
  store a gas's reference pressure. A species saved under the flag keeps 1 bar
  when read without it.
- **Fitted energy shifts behave like CatMap energies.** Because transition
  states ignore `dE` under the flag, a fitted shift acts like an energy change
  in CatMap, and thermodynamic rate control holds transition states fixed
  (Campbell's definition).

## The individual options

| CatMap | micki |
|---|---|
| Gas free energies at 1 bar, pressures in bar | `Gas(..., pref=1)`; `micki.utils.bar_to_molar(p, T)` for initial conditions |
| Transition state raised to max(IS, FS, TS) at the current coverages | `Reaction(..., clip='coverage')` |
| Step without a transition state (barrierless, kT/h prefactor) | `method='EQUIL', clip='coverage'` |
| Non-activated adsorption (collision-theory prefactor) | `method='STICK', clip='coverage'` |
| First-order interactions, `linear`/`piecewise_linear`/`smooth_piecewise_linear` response | `micki.lateral.first_order(adsorbates, eps, response=...)` |
| `transition_state_cross_interaction_mode`: `initial_state`, `intermediate_state`, `final_state`, `intermediate_state(w)` | `alpha=0`, `0.5`, `1`, `w` |
| Explicit transition-state interaction parameters | TS rows in `first_order`, `Reaction(..., explicit_ts=True)` |
| No configurational entropy | no `Model.lattice` |
| Fixed gas pressures | `Model.set_fixed([...gases])` with a CSTR |

## Porting a CatMap model

`tests/catmap_models.py` (function `build`) is a complete template. It takes a
model written in CatMap's input terms (formation energies, frequencies in
cm⁻¹, reaction expressions such as `'N2_g + 2*_s <-> N-N_s + *_s -> 2N_s'`)
and builds it in micki:

- **Gases:** `Gas` with the ASE geometry CatMap uses (`ase.build.molecule`),
  CatMap's symmetry number and spin (its `ideal_gas_params`), and the
  frequencies padded with zeros to 3N modes.
- **Adsorbates:** `Adsorbate` with the formation energy and frequencies. A
  transition state gets a placeholder first (imaginary) frequency.
- **Empty sites:** one per site type, energy 0.
- **Rate laws:** steps with a transition state are `TST`, steps without one
  `EQUIL`, and non-activated ones `STICK` (with `Model(..., Asite=)` from
  CatMap's `A_site` in Å² × 1e-20).
- **Reaction expressions:** `Reaction.from_string` reads CatMap's
  `'IS <-> TS -> FS'` form directly, given a dict that maps CatMap's names
  (including `*_s` for empty sites) to micki species.
- **Interactions:** `first_order` with CatMap's adsorbate interaction matrix.

Use CatMap's own numbers where it computes them itself rather than reading
your inputs:
- **Interaction cross terms:** CatMap fills them in with its
  `cross_interaction_mode` (geometric mean by default). The matrix it uses is
  in `model.thermodynamics.adsorbate_interactions._interaction_matrix` after a
  run.
- **Descriptor scaling:** with its usual scaler, CatMap fits adsorbate and
  transition-state energies, and even interaction parameters, to its
  descriptors. Take the free energies it reports (`scaler.get_free_energies`)
  or use its `ThermodynamicScaler`.

**Remaining differences** (CatMap behavior that micki does not reproduce):
- **Constants:** CatMap hard-codes CODATA-2010 kB and h (a ~5e-6 effect on
  rates) and, in its collision-theory prefactor, the atomic mass unit and the
  electron volt (5.3e-8 on those rate constants).
- **Site balance:** a multidentate species occupies one site in CatMap's
  balance, all its sites in micki's.
- **TOFs:** CatMap's gas turnover frequencies ignore stoichiometry (a gas
  appearing as `2X_g` counts once). Compare per-step rates instead.

## Validation

The test suite compares micki with CatMap 0.3.1 solving the same models
(`tests/test_catmap.py`):

- **Water-gas shift on Pt** (`tests/catmap_wgs.py`): the published model in
  CatMap's conventions, built under the flag, in two variants (linear
  response with α = 1/2; piecewise-linear response with explicit
  transition-state interactions), 4 conditions each. CO₂ rates and coverages
  agree to ~5e-14.
- **Ammonia synthesis** (Ru-like, non-activated H₂ adsorption, linear
  interactions) and **ethylene hydrogenation** (H on its own site type,
  interactions within and across site types, smooth piecewise-linear
  response) (`tests/catmap_models.py`), 3 conditions each, solved natively by
  CatMap with its own thermochemistry (ideal gas, harmonic adsorbates).
  Agreement: free energies ≤ 6e-16 eV, rate constants ~1e-14 (after dividing
  out the CODATA-2010 constants above), coverages ≤ 4e-13, rates ≤ 4e-8
  (limited by CatMap's integrated ammonia solution).

CatMap's CO-oxidation tutorial model (Pt(111), with and without its
interaction tutorial's smooth piecewise-linear interactions) gave the same
agreement in a one-off comparison: free energies ~1e-15 eV, rate constants,
coverages and rates ~1e-14.

CatMap's Newton solvers converge for some of these models but not for others
(ammonia synthesis, the interacting water-gas shift). The reference solutions
then come from integrating CatMap's own rate equations to steady state
(`tests/catmap_reference.py`, `tests/catmap_models_reference.py`).
