# Reference

Every public class and function with all of its parameters. Units are listed
in the [overview](index.md#units). Defaults marked *(conventions)* change under
`micki.set_conventions('catmap')` (see [CatMap conventions](catmap.md)).

- [Species](#species): `Gas`, `Liquid`, `Adsorbate`, `Electron`, common methods
- [Reaction](#reaction)
- [Model](#model)
- [ModelAnalysis](#modelanalysis)
- [Lattice](#lattice)
- [Lateral interactions: first_order](#microkilateralfirst_order)
- [Conventions](#microkiconventions)
- [EnergyReference](#energyreference), [databases](#microkidb), [VASP output](#microkioparse_vasp_out), [utilities](#microkiutils)

## Species

### Gas

```python
Gas(atoms, label, freqs=None, symm=1, spin=0., eref=None, rhoref=None,
    dE=0., pref=None, E=None, S=None, charge=0)
```

| Parameter | Default | Meaning |
|---|---|---|
| `atoms` | — | `ase.Atoms` with a calculator giving the potential energy (eV), an `ase.db` row, or the path of a VASP `OUTCAR`/`vasprun.xml` (energy and frequencies read from it). The geometry gives the moments of inertia. |
| `label` | — | Name; also the name of the sympy symbol for its concentration. |
| `freqs` | from `atoms` (row or VASP) | All 3N frequencies in eV, ascending; the lowest 6 (5 if linear) are dropped. |
| `symm` | 1 | Rotational symmetry number. |
| `spin` | 0 | Total spin S (electronic degeneracy 2S + 1). |
| `eref` | `None` | `EnergyReference` subtracted from the potential energy. |
| `rhoref` | 1 M | Reference concentration (M) of the free energy. |
| `dE` | 0 | Energy shift (eV). |
| `pref` | `None` *(conventions: 1)* | Reference pressure (bar) instead of `rhoref`; the reference concentration then follows T. |
| `E`, `S` | `None` | Given energy (eV) and entropy (eV/K) at the reference state, instead of computing them from the structure and frequencies (see below). |
| `charge` | 0 | Charge in units of e (charge balance of reactions). |

**Given energies.** With `E=` (and optionally `S=`, default 0), a species'
thermochemistry is not computed but taken as E and S, independent of
temperature: G(T) = E − T·S, plus kT (the ideal gas's pV) for gases only.
Solution-phase values are already enthalpies or free energies. The structure
is then used only for the composition (e.g. `Atoms('OOH')`, no calculator
needed), and `eref` is not applied. `dE`, lateral interactions and (for
adsorbates) configurational entropy still add. A plain free energy is E = G,
S = 0. The same holds for `Liquid` and `Adsorbate`.

Masses are the most abundant isotopes' (`micki.masses`), or ASE's standard
atomic weights *(conventions)*. Linear molecules are detected from the
geometry.

### Liquid

```python
Liquid(atoms, label, freqs=None, symm=1, spin=0., eref=None, rhoref=1.,
       S=None, D=None, dE=0., E=None, charge=0)
```

Same as `Gas` (without `pref`; a given E gets no kT), plus:

| Parameter | Default | Meaning |
|---|---|---|
| `D` | `None` | Diffusion coefficient (m²/s) for `DIFF` and `DIFF_LIQ`. |

### Solute

```python
Solute(label, E, formula, S=0., charge=0, rhoref=1., D=None, dE=0.)
```

A dissolved ion or molecule with a given free energy (a `Liquid` with given E
and S). Examples are H₃O⁺, OH⁻, acetate, or water as the solvent
(`rhoref=55.5`, so the pure liquid has activity 1). `formula` gives the
composition for the atom balance, e.g. `'H3O'`. G = E − T·S at the reference
concentration `rhoref` (M).

`liquid.R` is the average van der Waals radius (Å), used by `DIFF_LIQ`.

### Adsorbate

```python
Adsorbate(atoms, label, freqs=None, ts=None, spin=0., sites=None,
          lattice=None, eref=None, dE=0., symm=1, sitefree=False, E=None,
          S=None, charge=0)
```

| Parameter | Default | Meaning |
|---|---|---|
| `atoms` | — | As for `Gas`. An empty site can be `Atoms()` with energy 0, or the clean slab. |
| `label` | — | Name; also the name of its coverage symbol. |
| `freqs` | from `atoms` (row or VASP) | Frequencies (eV), all used; for a transition state the first (imaginary) one is dropped. |
| `ts` | `None` (false) | Transition state. |
| `spin` | 0 | Total spin S. |
| `sites` | `[]` | Site (vacancy) species occupied, once per site: `[slab]`, `[slab, slab]`. A species listed here by others is an empty site; empty sites have no sites themselves. |
| `lattice` | `None` | Set by `Model.lattice`; gives multi-site species configurational entropy. |
| `eref` | `None` | `EnergyReference`. |
| `dE` | 0 | Energy shift (eV). |
| `symm` | 1 | Symmetry number dividing the orientations counted by the lattice (2 for end-to-end symmetric multidentate species). Warns if larger than the number counted. |
| `sitefree` | `False` | The species deliberately occupies no sites (e.g. a mobile precursor). Otherwise a model raises an error for an adsorbate without sites that is not an empty site, and, with a lattice, for a transition state without sites. |
| `E`, `S` | `None` | Given energy and entropy instead of computed ones (see `Gas`); configurational entropy from a lattice still adds. |
| `charge` | 0 | Charge in units of e. |

### Electron

```python
Electron(label='e', dE=0.)
```

An electron at the electrode, for electrochemical steps: G = −e·U + dE, with U
the model's electrode potential (`Model.U_SHE`, V vs SHE). It has charge −1 and no
atoms. It has no concentration either: it does not enter rate laws as a
concentration and is not a variable. A reaction's `n_electrons` counts the
electrons it consumes.

### Common attributes and methods of species

| Name | Meaning |
|---|---|
| `label`, `symbol` | Name and sympy symbol (concentration or coverage in rate expressions). |
| `dE` | Energy shift (eV); can be changed at any time. |
| `lateral` | Coverage-dependent energy (eV), a number or sympy expression in symbols. |
| `scale` | Multipliers `{'E': {mode: x}, 'S': {mode: x}, 'H': x}` with modes `'tot'`, `'trans'`, `'trans2D'`, `'rot'`, `'vib'`, `'elec'`. |
| `conventions` | Conventions in effect when built. |
| `get_G(T)`, `get_H(T)`, `get_S(T)`, `get_E(T)`, `get_q(T)` | Free energy (eV; chemical potential at the reference state for fluids, Helmholtz energy for adsorbates), enthalpy, entropy (eV/K), internal energy, partition function. Include `lateral`. |
| `update(T=None, force=False)` | Recompute at T (cached otherwise). |
| `copy(newlabel=None)` | Copy, optionally renamed; keeps the conventions. |
| `save_to_db(db)` | Write to an ASE database (file name or connection). |
| `get_reference_state()` | Reference concentration (M) for fluids, 1 otherwise. |
| `+`, `*` | Build reaction sides: `2 * h + o2`. |

## Reaction

Reactions are usually written as strings (recommended):

```python
Reaction.from_string(expression, species, **kwargs)
reactions_from_strings(species, reactions)
```

- **`expression`:** `'reactants -> products'` or `'reactants <-> ts ->
  products'`.
  - `->` and `<->` are interchangeable; steps are reversible unless
    `reversible=False`.
  - Terms are labels joined by `+`, with optional integer coefficients (`2 h`,
    `2*h`, `2h`).
  - `*` is the empty site when the species include only one site type.
  - A term that is a key of `species` is always taken as that key.
  - Empty sites are optional, and ignored in the transition-state part.
- **`species`:** a dict `{label or other name: species}` (e.g. from
  `read_from_db`) or a list of species.
- **Keyword arguments:** those of the constructor below.
- **`reactions_from_strings`:** `reactions` maps names to an expression or to
  `(expression, {keyword arguments})`. Returns `{name: Reaction}` for
  `Model.add_reactions`; errors name the reaction.

The constructor takes the sides as species objects:

```python
Reaction(reactants, products, ts=None, method=None, S0=1., dG_act=None,
         dground=False, reversible=True, clip=..., alpha=...,
         explicit_ts=False, check_balance=True)
```

| Parameter | Default | Meaning |
|---|---|---|
| `reactants`, `products` | — | Species or sums of species. Missing empty sites are added to balance the sites. |
| `ts` | `None` | Transition-state species (or sum). |
| `method` | `'TST'` with `ts`/`dG_act`, else `'EQUIL'` | Rate law: `'TST'`, `'EQUIL'`, `'DIEQUIL'`, `'STICK'`, `'ER'`, `'DIFF'`, `'DIFF_LIQ'` (see the [user guide](user-guide.md#rate-laws)). Case-insensitive. |
| `S0` | 1 | Sticking coefficient (`'ER'`). |
| `dG_act` | `None` | Activation free energy (eV) instead of `ts`. |
| `dground` | `False` | Removed: `True` raises `ValueError`; use `clip='zero_coverage'`. |
| `reversible` | `True` | `False` omits the reverse rate. |
| `clip` | `None` *(conventions: `'coverage'` for TST, EQUIL, STICK)* | Negative barriers: `None` raises at zero coverage; `'zero_coverage'` (TST) clips once at zero coverage; `'coverage'` (TST, EQUIL, STICK, `dG_act`) uses Max(ΔG‡, ΔG, 0) at the current coverages. |
| `alpha` | `None` = computed *(conventions: 0.5)* | Weight of the products' coverage terms in the transition-state energy; computed BEP-style if `None`. |
| `explicit_ts` | `False` | Transition-state energy from the TS species only (with `ts.lateral`). |
| `check_balance` | `True` | Raise `ValueError` unless reactants, products and transition state contain the same atoms and carry the same charge (for the atoms, empty sites, i.e. species listed in a reaction species' `sites`, and electrons are not counted; an adsorbate's site atoms, e.g. the slab in its structure, removed). |

Under the CatMap conventions, transition states also ignore the `dE` of the
reactants and products (`reaction.ts_follows_dE = False`). An explicit
`clip=None` or `alpha=None` overrides the conventions' defaults.

| Attribute or method | Meaning |
|---|---|
| `keq`, `kfor`, `krev` | Equilibrium and rate constants (sympy expressions in coverages with lateral interactions). Rate constants are per site and second, in M⁻ⁿ for n fluid reactants. |
| `dG`, `dH`, `dS` | Reaction free energy, enthalpy, entropy (reference states of the species). |
| `dG_act`, `dH_act`, `dS_act` | Activation free energy, enthalpy, entropy. |
| `alpha` | α in use. |
| `n_electrons` | Electrons consumed as written (`Electron` species among the reactants minus among the products; positive for a reduction, 0 for a thermal step). |
| `method`, `clip`, `alpha_fixed`, `explicit_ts`, `ts_follows_dE`, `conventions` | Settings. |
| `scale`, `get_scale(p)`, `set_scale(p, x)` | Multipliers of `'dH_act'`, `'dS_act'`, `'kfor'`, `'krev'`. |
| `update(T, Asite, L, force=False)` | Recompute (called by `Model`). |
| `get_keq()`, `get_kfor()`, `get_krev()` | Same arguments as `update`. |

## Model

```python
Model(T, Asite, z=0, lattice=None, reactor='CSTR', rhocat=1,
      analytic_jac=False, U_SHE=0., pH=None)
```

| Parameter | Default | Meaning |
|---|---|---|
| `T` | — | Temperature (K). |
| `Asite` | — | Area of one site (m²); used by `STICK`, `ER`, `DIFF`. |
| `z` | 0 | Diffusion length (m) for `DIFF`. |
| `lattice` | `None` | `Lattice` or its neighbor dict, e.g. `{slab: {slab: 6}}`. Warns under the CatMap conventions. |
| `reactor` | `'CSTR'` | `'CSTR'` (all variables differential) or `'PFR'` (adsorbates algebraic). |
| `rhocat` | 1 | Site concentration (mol/L) converting per-site rates into fluid concentration changes. |
| `analytic_jac` | `False` | Exact (complex-step) Jacobian for IDA instead of difference quotients. |
| `U_SHE` | 0 | Electrode potential (V vs SHE), the potential of the `Electron` species. |
| `pH` | `None` | pH, used only to convert between the SHE and RHE scales; set the concentrations of H₃O⁺ and other species in the initial conditions. |

`T`, `Asite`, `z`, `lattice` and `U_SHE` are properties; setting them
rebuilds an initialized model. `U_RHE` (V vs RHE, U_SHE + (kT ln 10/e)·pH) is
read-only.
`set_potential(U, scale='SHE')` sets the potential on the `'SHE'` or `'RHE'`
scale.

| Method | Meaning |
|---|---|
| `add_reactions({name: Reaction})` | Add reactions and their species. Mixed conventions raise `ValueError`. |
| `set_fixed(labels)` | Keep these species (label or list) at their initial values. |
| `set_solvent(label)` | Make a `Liquid` the solvent (fixed). |
| `set_initial_conditions(U0)` | `{label: value}`: fluids in M, adsorbates as coverages; others 0, empty sites from the site balance. Builds the equations and the solver; raises `ValueError` for unknown species (also in lateral interactions), over-full sites, or adsorbates (with a lattice also transition states) that occupy no sites without `sitefree=True`. |
| `find_steady_state(dt=60, maxiter=2000, epsilon=1e-8, method='hybrid')` | Steady state; returns `(t, U, r)`. `'hybrid'`: Newton, else integrate (steps of `dt` s, at most `maxiter`, until max \|dy/dt\| < `epsilon`) and polish with Newton. `'integrate'`: integration only. `t` is `inf` after Newton alone. |
| `solve(t, ncp)` | Integrate to `t` (s) with `ncp` output points; returns `(U_list, r_list)`, times in `model.t`. |
| `copy(initialize=True)` | New model with the same settings and (shared) reactions. |
| `check_rates(U, epsilon=1e-6)` | Warn about rate constants above kT/h (run after every solve). |
| `finalize()` | Mark as not initialized. |

| Attribute | Meaning |
|---|---|
| `U`, `r`, `t` | Results of the last solve (lists of dicts, times). |
| `dU` | Time derivatives at the output points. |
| `steady_state_method` | `'newton'`, `'integrate+newton'` or `'integrate'`. |
| `reactions`, `species`, `vacancy`, `fixed`, `solvent` | Model contents. |
| `rates`, `dypdr` | Symbolic net rates and stoichiometry (after `set_initial_conditions`). |
| `_solver` | `micki.solver.IDASolver`: `residual(y, yp)`, `jacobian(y, cj)`, `rates(y)`, `newton(y)` for debugging. |

## ModelAnalysis

```python
ModelAnalysis(model, product_reaction, Uequil, tol=1e-3, dt=3600)
```

| Parameter | Default | Meaning |
|---|---|---|
| `model` | — | Model with reactions and fixed species. |
| `product_reaction` | — | Name of the reaction whose net rate r is analyzed. |
| `Uequil` | — | Initial conditions; the reference steady state is found from them. |
| `tol` | 1e-3 | Tolerance of `check_converged`. |
| `dt` | 3600 | Unused. |

| Method | Returns |
|---|---|
| `campbell_rate_control(rxn_name, scale=0.001)` | Degree of rate control (k/r) ∂r/∂k (forward and reverse scaled together). |
| `thermodynamic_rate_control(names, dg=None)` | −(kT/r) ∂r/∂G of the species `names` shifted together (default dg = 0.001 kT). |
| `activation_barrier(dT=0.01)` | Apparent activation energy kT² ∂ln r/∂T (eV). |
| `rate_order(name, drho=0.05)` | ∂ln r/∂ln c of species `name` (relative step `drho`). |
| `drate_order_dg(fluid, adsorbates, rho_scale=0.01, g_scale=0.01)` | Change of the rate order in `fluid` with the free energy of `adsorbates`. |
| `check_converged(*vals)` | Raise if the last two entries differ by more than `tol`. |

See [Sensitivity analysis](analysis.md).

## Lattice

```python
Lattice(neighborlist)
```

`neighborlist`: `{site: {neighbor site: count}}`, with the vacancy species (or
their labels) as keys. `ratio` gives the relative amounts of the site types;
`get_S_conf(sites)` gives the configurational entropy of a species on `sites`;
`update_site_names({label: species})` replaces labels by species.

## micki.lateral.first_order

```python
first_order(adsorbates, eps, response='linear', slope=1., cutoff=0.25,
            smoothing=0.05)
```

Sets `i.lateral = Σ_j F(θ_tot(j)) · eps[i][j] · θ_j` for every species `i` in
`eps` (CatMap's first-order model). `θ_tot(j)` is the total coverage of
`j`'s site type over the non-TS species in `adsorbates`.

| Parameter | Meaning |
|---|---|
| `adsorbates` | All adsorbates of the model (they define θ_tot). |
| `eps` | `{species_i: {species_j: eV}}`; rows may be transition states (use with `explicit_ts=True`). |
| `response` | `'linear'` (F = slope), `'piecewise_linear'` (0 up to `cutoff`, then slope·(θ − cutoff)/θ), `'smooth_piecewise_linear'` (quadratic between cutoff ∓ smoothing). |
| `slope`, `cutoff`, `smoothing` | Response parameters (CatMap's defaults). |

## micki.conventions

| Function | Meaning |
|---|---|
| `set_conventions(name)` | `'micki'` (default) or `'catmap'` for the rest of the program (context). |
| `conventions(name)` | Context manager: `with micki.conventions('catmap'): ...` |
| `get_conventions()` | Current setting. |

## EnergyReference

```python
EnergyReference(species, index=0)
```

Per-element reference energies from N structures with N elements (species,
`Atoms`, database rows or file names read with ASE at `index`); a read-only
dict `{element: eV}`. Pass it as `eref` to species, or use
`read_from_db(..., eref=labels)`.

## micki.db

| Function | Meaning |
|---|---|
| `read_from_db(db, names=None, eref=None)` | `{label: species}` from an ASE database (file name or connection); `names` selects labels, `eref` lists the labels defining the energy reference. Unparsable rows are skipped with a warning. |
| `row_to_thermo(row)` | One species from a row (sites still labels). |
| `species.save_to_db(db)` | Write a species (see above). |

## micki.io.parse_vasp_out

```python
parse_vasp_out(filename, ignore_atoms=())
```

`(atoms, freqs)` from a VASP `OUTCAR` or `vasprun.xml` frequency calculation:
the Hessian is diagonalized with micki's masses; `freqs` in eV, ascending,
imaginary modes negative; `ignore_atoms` (indices, atoms or element symbols)
are left out (partial Hessian).

## micki.utils

| Function | Meaning |
|---|---|
| `bar_to_molar(p, T)` | Ideal-gas concentration (M) at `p` bar and `T` K. |
| `calculate_avg_vdw_radius(atoms, npoints=8001)` | Average van der Waals radius (Å). |
