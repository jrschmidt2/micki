# User guide

A micki model is built in four steps: **species** (energies and vibrational
frequencies), **reactions** (elementary steps between species), a **model**
(reactions plus reactor settings), and a **solver** call. The
[reference](reference.md) lists every parameter; this guide explains how they
fit together. The theory is derived in Hermes et al., J. Chem. Phys. 151,
014112 (2019) (see the [overview](index.md)).

## A complete example

CO oxidation by dissociatively adsorbed O₂, at fixed gas pressures, with
made-up energies (eV, relative to the gases and the clean surface) and
frequencies (eV). The species are built directly from ASE `Atoms` objects with
attached energies; in practice they usually come from DFT output or an ASE
database (see below).

```python
from ase import Atoms
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator

from micki import Gas, Adsorbate, Reaction, Model
from micki.utils import bar_to_molar


def with_energy(atoms, energy):
    """Attach a potential energy (eV) to an ASE Atoms object."""
    atoms.calc = SinglePointCalculator(atoms, energy=energy)
    return atoms


# Gases: all 3N modes; the lowest 6 (5 for linear molecules) are dropped.
co_g = Gas(with_energy(molecule('CO'), 0.0), 'co_g',
           freqs=[0, 0, 0, 0, 0, 0.266])
o_g = Gas(with_energy(molecule('O2'), 0.0), 'o2_g',
          freqs=[0, 0, 0, 0, 0, 0.196], symm=2, spin=1)
co2_g = Gas(with_energy(molecule('CO2'), -3.0), 'co2_g',
            freqs=[0, 0, 0, 0, 0, 0.083, 0.083, 0.165, 0.291], symm=2)
# The empty site, and adsorbates occupying it
slab = Adsorbate(with_energy(Atoms(), 0.0), 'slab', freqs=[])
co = Adsorbate(with_energy(Atoms('CO'), -1.30), 'co',
               freqs=[0.254, 0.052, 0.045, 0.045, 0.008, 0.008],
               sites=[slab])
o = Adsorbate(with_energy(Atoms('O'), -1.20), 'o',
              freqs=[0.060, 0.051, 0.051], sites=[slab])
# Transition states: the first (imaginary) mode is dropped
o_o = Adsorbate(with_energy(Atoms('O2'), -0.70), 'o-o',
                freqs=[-0.05, 0.075, 0.057, 0.051, 0.037, 0.032],
                ts=True, sites=[slab, slab])
o_co = Adsorbate(with_energy(Atoms('CO2'), -1.70), 'o-co',
                 freqs=[-0.04, 0.22, 0.065, 0.053, 0.045, 0.037, 0.032,
                        0.015, 0.011],
                 ts=True, sites=[slab, slab])

reactions = {
    'co_ads': Reaction(co_g, co, method='STICK'),
    'o2_ads': Reaction(o_g, 2 * o, ts=o_o),
    'co_ox': Reaction(co + o, co2_g, ts=o_co),
}

T = 500.
model = Model(T, Asite=7e-20, reactor='CSTR')
model.add_reactions(reactions)
model.set_fixed(['co_g', 'o2_g', 'co2_g'])
model.set_initial_conditions({'co_g': bar_to_molar(0.1, T),
                              'o2_g': bar_to_molar(0.2, T),
                              'co2_g': 0.})
t, U, r = model.find_steady_state()
print('coverages: CO %.3f, O %.3g, free %.3f' % (U['co'], U['o'], U['slab']))
print('CO2 formation rate: %.3g per site per s' % r['co_ox'])
print('found by:', model.steady_state_method)
```

```
coverages: CO 0.117, O 0.882, free 0.001
CO2 formation rate: 1.21e+04 per site per s
found by: integrate+newton
```

The sections below go through each step.

## Species

| Class | Thermochemistry | State variable |
|---|---|---|
| `Gas` | ideal-gas translation, rigid rotor, harmonic vibrations | concentration (M) |
| `Liquid` | as `Gas`, at a reference concentration; diffusion coefficient for diffusion rate laws | concentration (M) |
| `Adsorbate` | harmonic vibrations only (adsorbates, transition states, empty sites) | coverage |
| `Electron` | energy only (electrochemistry) | — |

### Energies

The electronic energy of a species is the potential energy of its `atoms`
(from the attached ASE calculator), minus an optional per-element reference
`eref`, plus an optional shift `dE`:

- **`eref`**: a `micki.EnergyReference` built from N structures containing N
  elements (for example the clean slab, CO, H₂O and H₂), which gives every
  element a reference energy so that formation energies are referenced
  consistently. `read_from_db(..., eref=[labels])` sets it for all species read.
- **`dE`**: a constant shift in eV, for corrections (e.g. to match an
  experimental binding energy) or fitting parameters. Transition states follow
  the `dE` of their reactants and products through α (see
  [Transition states](#transition-states-and-α)).

`get_G(T)`, `get_H(T)`, `get_S(T)`, `get_E(T)` and `get_q(T)` return the
thermodynamic functions. For fluids, G is the chemical potential at the
reference state (including pV = kT); for adsorbates it is the Helmholtz energy.

### Frequencies

Frequencies are in eV, and imaginary modes are negative numbers.

- **Gases and liquids:** give all 3N modes sorted ascending, as from a full
  Hessian. The lowest 6 (5 for linear molecules) are the translations and
  rotations and are dropped; zeros are fine as placeholders.
- **Adsorbates:** all given frequencies are used. They must be positive.
- **Transition states (`ts=True`):** the first frequency, the imaginary mode,
  is dropped.

### Reading DFT output and databases

`atoms` can also be the path of a VASP `OUTCAR` or `vasprun.xml` from a
frequency calculation. micki then reads the structure and energy from it and
diagonalizes the Hessian itself (`micki.io.parse_vasp_out`), so all species use
the same masses and unit conversions.

Species can be stored in an ASE database with `species.save_to_db(db)` and read
back with `micki.db.read_from_db`:

```python
from micki.db import read_from_db

sp = read_from_db('species.json', eref=['slab', 'co_g', 'h2o_g', 'h2_g'])
sp['co'].dE = 0.095          # e.g. correct the CO binding energy
```

The database row stores the frequencies, sites (as labels), `ts`, `symm`,
`spin`, `dE`, the reference state, and `D` and `S` for liquids. Energy
references, lattices and lateral interactions are not stored: set them again
after reading.

### Sites and multidentate species

Empty sites are species too: an `Adsorbate` without sites (e.g. the clean slab
with energy 0 relative to the reference). Every adsorbate lists the sites it
occupies, once per site:

```python
co = Adsorbate(atoms_co, 'co', freqs_co, sites=[slab])            # monodentate
ts = Adsorbate(atoms_ts, 'o-co', freqs_ts, ts=True, sites=[slab, slab])
```

The coverages of all species on a site type plus its empty sites add up to 1.
With several site types, the lattice sets their ratio. Several site types work
as you would expect: give each its own vacancy species (e.g. H on `hollow`,
hydrocarbons on `top`).

### Gas reference state

The free energy of a gas refers to a concentration of 1 M by default. Use
`rhoref=` for another concentration (M), or `pref=` for a pressure in bar,
e.g. `pref=1` (CatMap's and ASE's convention). Rates and steady states don't
depend on the reference state. Reported free energies, ΔG and keq do, and so do
the computed α and clipped barriers.

## Reactions

```python
Reaction(reactants, products, ts=None, method=None, ...)
```

Reaction sides are species combined with `+` and `*`: `2 * o`, `co + o`,
`sp['cooh'] + sp['slab']`. Missing empty sites are added automatically to
balance the sites on both sides. In the example, `co_g -> co` becomes
`co_g + slab -> co`.

### Rate laws

`method` selects how the forward rate constant is computed. The reverse rate
constant is always `kfor / keq`, so every rate law obeys detailed balance.

| method | Rate constant | Typical use |
|---|---|---|
| `'TST'` (default with `ts` or `dG_act`) | kT/h · exp(−ΔG‡/kT) | surface reactions with a transition state |
| `'EQUIL'` (default otherwise) | kT/h, times keq if keq < 1 at zero coverage | fast steps without a barrier |
| `'DIEQUIL'` | 1/kfor = h/kT · (1 + 1/keq), a smooth version of `EQUIL` | as `EQUIL` |
| `'STICK'` | collision theory, sticking coefficient 1 (TST with a 2D ideal gas transition state) | non-activated adsorption of one fluid |
| `'ER'` | collision theory with sticking coefficient `S0` | Eley–Rideal reactions |
| `'DIFF'` | diffusion of a fluid (coefficient `D`) through a layer of thickness `Model.z` | mass-transfer limits in liquids |
| `'DIFF_LIQ'` | diffusion-limited (Smoluchowski) reaction of two liquids | solution-phase reactions |

`dG_act=` gives a TST barrier directly, without a transition-state species.
`reversible=False` drops the reverse rate.

### Transition states and α

A transition state's free energy follows the coverage-dependent energy terms
(lateral interactions and `dE`) of the reactants and products:

G‡(θ) = G‡ + (1 − α)·Σ_reactants (lateral + dE) + α·Σ_products (lateral + dE)

By default α is computed self-consistently from the forward and reverse
barriers, BEP-style (Hermes et al. 2019): an early transition state follows the
reactants, a late one the products. This is also how fitted `dE` shifts and
thermodynamic rate control reach the transition states. `alpha=w` fixes the
weight instead (0 = follow the reactants, 1 = follow the products), and
`explicit_ts=True` uses only the transition state's own energy, including
`ts.lateral`.

### Negative barriers

By default (`clip=None`), a negative forward or reverse barrier at zero
coverage, computed without `dE` and lateral terms, raises an error. `clip`
offers two alternatives:

- **`'zero_coverage'`** (TST): decides at zero coverage. A negative forward
  barrier becomes 0, and a negative reverse barrier makes the forward barrier
  ΔG. That choice is then kept at all coverages. (This replaces the former
  `dground=True`.)
- **`'coverage'`** (TST, EQUIL, STICK): uses the barrier Max(ΔG‡, ΔG, 0) at the
  current coverages, i.e. the transition state is raised to the higher of the
  initial and final states, as CatMap does.

### Rate constants and sensitivity

After `update` (which `Model` calls), `reaction.keq`, `kfor` and `krev` hold
the constants, as sympy expressions in the coverages when there are lateral
interactions. `dG`, `dH`, `dS` and `dG_act`, `dH_act`, `dS_act` hold the
reaction and activation thermochemistry. `reaction.set_scale(param, value)`
multiplies `'kfor'`, `'krev'`, `'dH_act'` or `'dS_act'`; the sensitivity
analysis uses it.

## Lateral interactions

Coverage-dependent energies go into `species.lateral` as sympy expressions in
the coverages, which are the species' `symbol`s. For example, CO–CO and CO–O
repulsion:

```python
co.lateral = 1.5 * co.symbol + 1.1 * o.symbol
o.lateral = 1.1 * co.symbol + 2.2 * o.symbol
```

- **Symmetric matrix:** the interaction matrix must be symmetric (the CO–O and
  O–CO terms equal) for the model to be thermodynamically consistent.
- **Transition states:** they follow through α, as above.
- **Any functional form works:** `Piecewise`, `Max` and `Min` are fine (the
  Jacobian handles them).
- **Unknown species are an error:** a lateral term referring to a species that
  is not in the model raises an error when the model is built.

`micki.lateral.first_order(adsorbates, eps, response=...)` writes CatMap's
first-order interaction model into `lateral`, with linear, piecewise-linear or
smooth piecewise-linear response functions (see [CatMap conventions](catmap.md)).

## Lattice and configurational entropy

```python
model.lattice = {slab: {slab: 6}}     # or micki.Lattice({...})
```

A lattice, given as neighbor counts between site types, gives species that
occupy several sites their configurational entropy, kB ln(number of
orientations). On a hexagonal lattice that is 6 for a bidentate species
(Hermes et al. 2019, eqs. 6–7).
- **Symmetry:** an end-to-end symmetric species should set `symm=2`, which
  divides the count.
- **Single-site species:** they have no orientations counted, so `symm > 1` on
  them only lowers the partition function, and micki warns.
- **Several site types:** the lattice also sets their relative amounts.

## Models

```python
model = Model(T, Asite, z=0, lattice=None, reactor='CSTR', rhocat=1,
              analytic_jac=False)
model.add_reactions({'name': reaction, ...})
model.set_fixed(['co_g', ...])
model.set_initial_conditions(U0)
```

- **`T` (K) and `Asite` (m², the area of one site):** collision-theory rate laws
  use `Asite`.
- **Changing settings:** changing `model.T`, `Asite`, `z` or `lattice` later
  rebuilds the model.
- **`set_fixed`:** keeps concentrations constant, e.g. gases at fixed partial
  pressure. A model with no fixed species has conserved quantities (atoms,
  sites), so its steady state is found by integration.
- **Initial conditions:** fluid concentrations in M (use
  `bar_to_molar(p, T)`), adsorbate coverages as fractions. Unlisted species
  start at 0, and empty sites follow from the site balances.
- **`reactor='CSTR'`:** all concentrations evolve in time, with fluids and
  surface coupled through `rhocat`, the concentration of sites (mol/L).
- **`reactor='PFR'`:** adsorbates are algebraic (pseudo-steady state), and
  `solve` integrates the fluid concentrations along the reactor, with time
  standing for residence time. The [water-gas shift example](examples/wgs.md)
  solves a CSTR steady state and uses it to start a PFR.
- **`set_solvent(label)`:** makes a `Liquid` the solvent, with a fixed
  concentration.

## Solving

```python
t, U, r = model.find_steady_state()     # steady state
U_list, r_list = model.solve(t, ncp)     # integrate to t (s), ncp outputs
```

`U` maps species labels, including empty sites, to concentrations or coverages.
`r` maps reaction names to net rates per site per second (forward minus
reverse).

**`find_steady_state`** (default `method='hybrid'`):
1. It first solves the steady-state equations directly with Newton's method in
   ln(y) and the exact Jacobian. That converges from a nearby guess, e.g. a
   previous steady state.
2. Otherwise it integrates in time (steps of `dt` seconds, at most `maxiter`
   steps) until max |dy/dt| < `epsilon`, then polishes with Newton.

`model.steady_state_method` records which path was taken: `'newton'`,
`'integrate+newton'` or `'integrate'`. `method='integrate'` only integrates.

**Jacobian:** IDA uses difference quotients for the Jacobian unless
`Model(..., analytic_jac=True)`, which supplies the exact complex-step Jacobian
(the steady-state Newton solver always uses it). Both give the same results.

**Rates near equilibrium:** the net rate of a nearly equilibrated step is a
small difference of large forward and reverse fluxes, so in double precision
it carries a relative error of about 1e-16 × flux/net rate. Read overall rates
from the slow steps (or from the change of the fluids).

**Warnings and errors:**
- After every solve, `check_rates` warns about rate constants above kT/h.
- The residual rejects negative concentrations.
- Over-full sites in the initial conditions raise an error.
