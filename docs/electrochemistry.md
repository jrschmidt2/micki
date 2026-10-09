# Electrochemistry

micki models electrochemical reactions at an electrode held at a fixed
potential:
- **Electron transfer:** proton-coupled electron transfer (PCET) and other
  electron-transfer steps, with potential-dependent barriers.
- **Proton donors:** any number of them (H₃O⁺, water, weak acids), each a
  separate reaction channel with its own concentration.
- **Transport:** of reactants, products and donors through a Nernst
  diffusion layer.
- **Outputs:** current densities, Faradaic efficiencies, selectivities,
  polarization curves and Tafel slopes.

The formalism follows the approach of Mondal, Sheng, Lee, Jin and Schmidt,
ACS Catal. 15, 8788 (2025), but it uses explicit solution species and mass
action throughout instead of the computational hydrogen electrode (see
[Why there is no CHE mode](#why-there-is-no-che-mode)).

## A complete example

Hydrogen evolution with two proton donors, H₃O⁺ and water, on a model
surface, with H₃O⁺ and OH⁻ transported to and from a rotating disk electrode.
As in the [user guide](user-guide.md#a-complete-example), the energies are made
up, so the example is self-contained. Here they are free energies relative to
H₂(g) and H₂O(l).

```python
from ase import Atoms

from micki import Adsorbate, Model, Solute, reactions_from_strings
from micki.electrochem import Electrolyte, levich_delta

# Solution species on the SHE scale, relative to H2(g) and H2O(l):
# h3o_aq (H3O+), h2o_l (water, activity 1), oh_aq (OH-) and the electron e
electrolyte = Electrolyte(G_H2=0., G_H2O=0.)
sp = electrolyte.species()
sp['h3o_aq'].D = 9.3e-9  # diffusion coefficients (m^2/s)
sp['oh_aq'].D = 5.3e-9
# near-surface copies, connected to the bulk by film transport
sp['h3o_dl'] = sp['h3o_aq'].copy('h3o_dl')
sp['oh_dl'] = sp['oh_aq'].copy('oh_dl')

slab = Adsorbate(Atoms(), 'slab', E=0.)            # the empty site
sp['slab'] = slab
sp['h'] = Adsorbate(Atoms('H'), 'h', E=0.1, sites=[slab])  # vs 1/2 H2
sp['h2_aq'] = Solute('h2_aq', 0., 'H2')            # dissolved H2

reactions = reactions_from_strings(sp, {
    # transport of H3O+ and OH- between the bulk and the electrode
    'film_h3o': ('h3o_aq -> h3o_dl', {'method': 'FILM'}),
    'film_oh': ('oh_aq -> oh_dl', {'method': 'FILM'}),
    # Volmer step with two proton donors; barrier at dG = 0, beta = 0.5
    'volmer_h3o': '* + h3o_dl + e <-> ^0.35 -> h + h2o_l',
    'volmer_h2o': '* + h2o_l + e <-> ^0.55 -> h + oh_dl',
    # Tafel recombination (thermal)
    'tafel': ('2 h <-> ^0.7 -> h2_aq + 2 *', {'beta': 0.5}),
})

for pH in (1, 3, 13):
    # rotating disk at 1600 rpm (water: nu = 1e-6 m^2/s)
    model = Model(298.15, Asite=7e-20, pH=pH,
                  delta=levich_delta(nu=1e-6, rpm=1600))
    model.add_reactions(reactions)
    model.set_fixed(['h3o_aq', 'h2o_l', 'oh_aq', 'h2_aq'])
    U0 = electrolyte.concentrations(pH)
    U0['h2_aq'] = 1e-6
    U0['h3o_dl'], U0['oh_dl'] = U0['h3o_aq'], U0['oh_aq']
    model.set_potential(-0.05, scale='RHE')
    model.set_initial_conditions(U0)
    result = model.sweep([-0.05, -0.2, -0.4, -0.6, -0.8], scale='RHE')
    for U, j, rates in zip(result.U, result.j, result.rates):
        partial = model.partial_currents(rates)
        print('pH %2d  U = %5.2f V vs RHE  j = %8.3g mA/cm2  '
              'from H3O+: %3.0f%%' % (pH, U, j,
                                       100 * partial['volmer_h3o'] / j))
```

```
pH  1  U = -0.05 V vs RHE  j =    -3.15 mA/cm2  from H3O+: 100%
pH  1  U = -0.20 V vs RHE  j =     -188 mA/cm2  from H3O+: 100%
pH  1  U = -0.40 V vs RHE  j =     -205 mA/cm2  from H3O+: 100%
pH  1  U = -0.60 V vs RHE  j =     -205 mA/cm2  from H3O+: 100%
pH  1  U = -0.80 V vs RHE  j =     -205 mA/cm2  from H3O+: 100%
pH  3  U = -0.05 V vs RHE  j =    -1.33 mA/cm2  from H3O+: 100%
pH  3  U = -0.20 V vs RHE  j =    -3.42 mA/cm2  from H3O+: 100%
pH  3  U = -0.40 V vs RHE  j =    -3.45 mA/cm2  from H3O+:  99%
pH  3  U = -0.60 V vs RHE  j =    -14.5 mA/cm2  from H3O+:  24%
pH  3  U = -0.80 V vs RHE  j =     -167 mA/cm2  from H3O+:   2%
pH 13  U = -0.05 V vs RHE  j =    -2.92 mA/cm2  from H3O+:   0%
pH 13  U = -0.20 V vs RHE  j =     -156 mA/cm2  from H3O+:   0%
pH 13  U = -0.40 V vs RHE  j =     -204 mA/cm2  from H3O+:   0%
pH 13  U = -0.60 V vs RHE  j =     -205 mA/cm2  from H3O+:   0%
pH 13  U = -0.80 V vs RHE  j =     -205 mA/cm2  from H3O+:   0%
```

What the results show:
- **pH 1:** H₃O⁺ carries the whole current.
- **pH 13:** water does.
- **pH 3, the plateau:** the current levels off at the H₃O⁺ diffusion limit,
  about 3.4 mA/cm² at 1 mM and 1600 rpm.
- **pH 3, the switch:** at more negative potentials, water reduction takes over.
- **The common ceiling:** about 205 mA/cm², set by the Tafel recombination
  step.

## Potentials and electrons

- **`Electron('e')`:** the electrode's electrons, with G = −eU + dE, where U
  is the electrode potential vs SHE. It has charge −1 and no concentration: it
  is never a variable and does not enter rate laws as a concentration.
- **`Model(..., U_SHE=..., pH=...)`:** `U_SHE` is a property like `T`;
  setting it updates the model. `pH` is used only to convert between scales.
  `U_RHE = U_SHE + (kT ln 10/e)·pH` is read-only.
- **`set_potential(U, scale='RHE')`:** sets the potential on either scale.
- **`n_electrons`:** each reaction's net number of electrons consumed as
  written, counted automatically: 1 for a one-electron reduction, −1 for an
  oxidation written forward, 0 for thermal steps.
- **Balance checks:** reactions check charge as well as atoms, so a forgotten
  `e` (or proton) is an error.

## Solution species

`micki.electrochem.Electrolyte(G_H2, G_H2O, pKw=14, acids={...})` provides
two things that are kept separate.

1. **Standard free energies** (`species()`), on the SHE scale:
   G(H₃O⁺, 1 M) + G(e⁻, 0 V) = ½G(H₂, 1 bar) + G(H₂O).
   - **Hydroxide:** G(OH⁻) follows from pKw.
   - **Conjugate bases:** each G(A⁻) follows from its pKa, via
     HA + H₂O ⇌ H₃O⁺ + A⁻ with ΔG = kT ln10·pKa.
   - **Consistency:** this makes the proton-transfer channels of all donors
     consistent with each other and with the electrode: they predict the same
     equilibria.
2. **Concentrations at a pH** (`concentrations(pH, totals={...})`):
   - [H₃O⁺] = 10⁻ᵖᴴ;
   - [OH⁻] = K_w/[H₃O⁺];
   - water at 55.5 M (activity 1);
   - each buffer split by Henderson–Hasselbalch.

   These are inputs for the initial (fixed) concentrations; no electrode
   equilibrium is involved.

Default labels are `h3o_aq`, `h2o_l`, `oh_aq` and `e`, chosen to avoid
clashes with adsorbate labels; `labels=` changes them. A weak acid is added
as, for example:

```python
electrolyte = Electrolyte(acids={'hoac': {
    'base': 'oac', 'pKa': 4.76, 'formula': 'C2H4O2',
    'base_formula': 'C2H3O2'}})
U0 = electrolyte.concentrations(5.0, totals={'hoac': 0.1})
```

Any solution species can also be defined directly as a
`Solute(label, E, formula, S=0., charge=0, rhoref=1., D=None)`. The given
energies E (and S) are used instead of computed thermochemistry; this works
for any species, see `E=`/`S=` in the [reference](reference.md#gas).

## Multiple proton donors

Each donor is a separate reaction, with its own barrier and β:

```python
'h_h3o':  '* + h3o_aq + e <-> ^0.35 -> h + h2o_l',
'h_h2o':  '* + h2o_l + e <-> ^0.55 -> h + oh_aq',
'h_hoac': '* + hoac + e <-> ^0.45 -> h + oac',
```

Which donor dominates follows from the model: its concentration enters the
rate through mass action, and its barrier through ΔG, which contains the
pKa. At fixed U_RHE, the forward rate of the H₃O⁺ channel changes by
10^(β−1) per pH unit, while that of the water channel changes by 10^β.

## PCET barriers

There are two ways to give the barrier of an electrochemical step.

- **A transition state, or `dG_act`, at a reference potential** (e.g. from
  constant-potential DFT), with `U_ref` in V vs SHE:

  ΔG‡(U) = ΔG‡(U_ref) + β·[ΔG(U) − ΔG(U_ref)]

  ```python
  Reaction.from_string('ooh + h3o_aq + e <-> ooh-ts -> o + 2 h2o_l', sp,
                       U_ref=0.6, beta=0.3)
  ```

- **A barrier at ΔG = 0** ("simple" PCET, e.g. 0.26 eV at the step's
  equilibrium potential):

  ΔG‡ = ΔG‡⁰ + β·ΔG

  This is CatMap's G_TS = G_FS + barrier + (1 − β)(−ΔG). In reaction strings
  it is written `'o2 + h3o_aq + e <-> ^0.26 -> ooh + h2o_l'`, or with
  `dG_act0=0.26`.

Details for both forms:
- **β applies to the step as written.** The reverse barrier changes with
  1 − β automatically (krev = kfor/keq). An oxidation written forward
  therefore needs β_ox = 1 − β_red.
- **Defaults:** β = 0.5. Electrochemical steps also default to
  `clip='coverage'`, because barriers extrapolated over a potential sweep
  would otherwise become negative. With a transition state they default to
  `alpha=β`. Explicit arguments win.
- **`dG_reorg`:** adds a barrier that computed barriers miss, such as solvent
  reorganization (e.g. 0.22 eV for PCET or 0.28 eV for adsorption/desorption
  in Mondal et al.). It multiplies the rate constant by exp(−dG_reorg/kT), for
  any rate law, including barrierless (`EQUIL`) steps.
- **`prefactor`:** replaces kT/h (e.g. to mirror CatMap models with
  `prefactor=1e9`).

## Mass transport

`method='FILM'` transports a species through a Nernst diffusion layer to its
near-surface copy (`'x_aq -> x_dl'`, with `x_dl = x_aq.copy('x_dl')`). The
rate constant per site is

  k = 1000·D·N_A·Asite/(roughness·δ)

from a flux D(c_bulk − c_near)/δ per geometric area.

- **The film thickness δ:** `Model(delta=...)`, or per reaction
  `Reaction(..., delta=...)`. It can be a function of D, such as
  `levich_delta(nu, rpm)`, the rotating-disk value
  1.61 D^⅓ ν^⅙ ω^−½, so that each species gets its own thickness.
- **`Model(roughness=...)`:** the electrochemically active area per geometric
  area (e.g. C_dl/C_s), which sets the sites per geometric area
  (roughness/Asite).
- **Near-surface species** change at per-site rates through `rhocat`, like
  surface steps. Steady states of transport and surface steps alone don't
  depend on `rhocat`. `film_rhocat(Asite, roughness, delta)` gives the
  physical value, needed for transients or for solution reactions in the film
  (e.g. buffer equilibria).
- **Limitations:**
  - There is no migration or electroneutrality (a supporting electrolyte is
    assumed).
  - Solution reactions in the film, such as H₃O⁺ + OH⁻ neutralization, act
    only if you add them as reactions.

  Local pH is therefore approximate.

## Results

| Method | Result |
|---|---|
| `model.current()` | Current density, −e·(roughness/Asite)·Σ n_electrons·r. In mA/cm², cathodic negative by default; `unit=`, `cathodic_negative=False`. |
| `model.partial_currents()` | `{reaction: j}`, e.g. per donor or per pathway. |
| `model.faradaic_efficiency(label, n)` | Fraction of the electrons that make `label`. |
| `model.selectivity(product, reactant)` | Product per reactant consumed. For oxygen reduction, `selectivity('h2o2', 'o2')` is the RRDE H₂O₂ selectivity 2·j_peroxide/(j_peroxide + j_total). |
| `model.sweep(potentials, scale='RHE')` | Steady states over potentials, each started from the previous one: `U`, `j`, `states`, `rates`, `method`. |
| `model.tafel_slope()`, `electrochem.tafel_slope(U, j)` | Tafel slopes (mV/decade) and apparent transfer coefficients. |
| `ModelAnalysis(model, 'current', U0)` | Degrees of rate control of the current (or of any function `f(model, rates)`). |

## Why there is no CHE mode

**What CHE does.** The computational hydrogen electrode lumps H⁺ and e⁻ into
one species with G = ½G(H₂) − eU_RHE. Its concentration does not appear in
rate laws, so all rates depend on pH and potential only through U_RHE. The
thermodynamics are exact. The kinetics are not: at fixed U_SHE, a CHE rate
scales as [H⁺]^β. With an explicit donor, the rate is first order in [H₃O⁺]
at fixed U_SHE, so the two agree only for β = 1.

**Why micki leaves it out.** micki always uses explicit species. Its
mass-action kinetics stay correct at any pH, for any β, and with any number
of donors.

**Translating CHE- or CatMap-based models.** A CatMap model whose proton–
electron pair `pe_g` has activity 1 (voltages vs RHE) corresponds to pH 0
here, where SHE and RHE coincide. Write each `pe_g` as `h3o_aq + e` with
H₃O⁺ at 1 M, plus a released `h2o_l` at activity 1. The test suite does
exactly this for CatMap's oxygen-reduction and hydrogen-evolution models:

```
'O2_a + pe_g <-> ^0.26eV_a -> OOH_a'      (CatMap)
'O2_a + h3o_aq + e <-> ^0.26 -> OOH_a + h2o_l'   (micki, pH 0)
```

## Validation

- **Analytic limits** (`tests/test_electrochem.py`):
  - Nernst shifts of 59.16 mV per pH unit;
  - Tafel slopes of 2.303kT/(βe);
  - Levich-limited fluxes;
  - buffer equilibria, and consistency between donors;
  - the pH dependence of donor channels;
  - currents, selectivities and sweeps.
- **CatMap** (`tests/test_catmap.py`): oxygen reduction (2e⁻/4e⁻ pathways,
  double-layer transport, β = 0.5 and 0.1, Hansen et al. 2014 energies) at 7
  potentials, and hydrogen evolution at 4. Rate constants and coverages agree
  to about 1e-14.
