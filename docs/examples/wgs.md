# Example: water-gas shift on Pt

The water-gas shift reaction, CO + H₂O → CO₂ + H₂, on Pt(111), modeled after
Hermes et al., J. Chem. Phys. 151, 014112 (2019) and compared with the
experiments of Grabow et al. (J. Phys. Chem. C 112, 4608 (2008), Table 5) at 21
reaction conditions. The complete script is
[`examples/wgs.py`](../../examples/wgs.py). Run it from a clone of the
repository:

```
python examples/wgs.py
```

The same model, written with species objects and operators instead of
reaction strings, is micki's regression test,
[`tests/wgs.py`](../../tests/wgs.py). A test checks that the two are identical.

## Species

DFT energies (PBE-D3) and frequencies are in an ASE database,
`tests/data/wgs.json`. It contains the gases, the adsorbates, the transition
states, and the empty site `slab`.
- **The empty site:** `slab` is the clean Pt(111) slab, an `Adsorbate` that
  the other adsorbates list as their site. Nothing else marks it as a site.
- **Energy reference:** it makes the slab, CO, H₂O and H₂ zero, so adsorbate
  energies are binding energies.

```python
sp = read_from_db(DATABASE, eref=['slab', 'co_g', 'h2o_g', 'h2_g'])

# The O-H-OH transition state is end-to-end symmetric.
sp['o-h-oh'].symm = 2
```

Following the paper, the CO and H binding energies are shifted to
experimental binding enthalpies. The OH and COOH shifts are fitted to the
measured turnover frequencies (least squares in ln TOF over all 21
conditions):

```python
sp['co'].dE = 0.09496182099234107
sp['h'].dE = 0.236689058 / 2
sp['oh'].dE = -0.2280
sp['cooh'].dE = 0.1603
```

Lateral interactions are linear in the coverages: CO with every adsorbate
(each pair entered symmetrically), and O with itself:

```python
co, o = sp['co'].symbol, sp['o'].symbol
pairs = {'co': 2 * 0.784423808, 'o': 1.147079243, 'h': 0.237728186,
         'h2o': 0.10555879, 'oh': 0.263160274, 'cooh': 1.901900269}
sp['co'].lateral = sum(eps * sp[name].symbol
                       for name, eps in pairs.items())
for name, eps in pairs.items():
    if name != 'co':
        sp[name].lateral = eps * co
sp['o'].lateral += 2 * 1.11913902 * o
```

## Reactions

```python
reactions = reactions_from_strings(sp, {
    'co_ads': ('co_g + * -> co', {'method': 'STICK'}),
    'h2o_ads': ('h2o_g + * -> h2o', {'method': 'STICK'}),
    'h2_ads': ('h2_g + 2* -> 2 h', {'method': 'STICK'}),
    'ho-h': 'h2o + * <-> ho-h -> oh + h',
    'o-h': 'oh + * <-> o-h -> o + h',
    'o-h-oh': ('2 oh -> o + h2o', {'method': 'DIEQUIL'}),
    'o-co': 'o + co <-> o-co -> co2_g + 2*',
    'co-oh': 'cooh + * <-> co-oh -> co + oh',
    'oco-h': 'cooh + * <-> oco-h -> co2_g + h',
    'oco-h-o': 'cooh + o -> co2_g + oh + *',
    'oco-h-oh': 'cooh + oh -> co2_g + h2o + *',
})
```

**Reading the expressions:**
- **Form:** each is `'reactants <-> transition state -> products'` or
  `'reactants -> products'`.
- **`*`:** stands for the empty site, `slab` here. Empty sites are written out
  for clarity, but they're optional: missing ones are added to balance the
  sites, so `'co_g -> co'` gives the same reaction.
- **Rate laws:** steps with a transition state use transition state theory, and
  steps without one are barrierless (`EQUIL`) unless a `method` is given.
  Adsorption here is non-activated (`STICK`, collision theory).
- **α:** transition states follow the lateral interactions and energy shifts
  of their reactants and products through the self-consistent α.
- **`o-h-oh`:** with `DIEQUIL`, the step has no transition state.

## Reactor model

Each condition is solved in two steps:

1. A CSTR at the inlet composition (fixed gases) is solved to steady state. It
   gives the surface coverages at the reactor inlet.
2. A plug-flow reactor started from that state is integrated over the
   residence time. The adsorbates are at pseudo-steady state (algebraic), and
   the gases change along the reactor.

```python
def turnover_frequency(condition, sp, reactions):
    """TOF (CO2 per site per minute) at a reaction condition."""
    T, p_co, p_h2o, p_co2, p_h2, flow, _ = condition
    inlet = {name: bar_to_molar(ATM * p, T)
             for name, p in zip(GASES, (p_co, p_h2o, p_co2, p_h2))}

    # surface at the reactor inlet: CSTR with fixed gases
    cstr = Model(T, ASITE, reactor='CSTR')
    cstr.lattice = {sp['slab']: {sp['slab']: 6}}
    cstr.add_reactions(reactions)
    cstr.set_fixed(GASES)
    cstr.set_initial_conditions({**inlet, **COVERAGE0})
    _, U_inlet, _ = cstr.find_steady_state(maxiter=200000)

    # plug-flow reactor over the residence time (s), adsorbates at
    # pseudo-steady state
    residence_time = 60 * 1000 * NSITES / flow
    pfr = Model(T, ASITE, reactor='PFR', rhocat=1.)
    pfr.lattice = cstr.lattice
    pfr.add_reactions(reactions)
    # (empty sites follow from the site balance)
    pfr.set_initial_conditions({name: value for name, value in U_inlet.items()
                                if name != 'slab'})
    U, _ = pfr.solve(residence_time, 1000)

    return (U[-1]['co2_g'] - inlet['co2_g']) * flow / (1000 * NSITES)
```

**Settings:**
- **Lattice:** hexagonal (6 neighbors per site). It gives the two-site
  transition states their configurational entropy.
- **`NSITES`:** the number of Pt sites in the experiment, from the Pt loading
  and dispersion.
- **`ASITE`:** the area of one Pt(111) site.

## Results

`python examples/wgs.py` prints, in about 5 s:

```
 #   T/K  p(CO) p(H2O) p(CO2) p(H2)   TOF/min  measured
 1   523  0.154  0.208  0.000 0.000      3.36      3.68
 2   548  0.055  0.208  0.000 0.000      8.08      8.56
 3   548  0.105  0.208  0.000 0.000      7.46      8.06
 4   548  0.137  0.062  0.000 0.000      2.98      3.63
 5   548  0.144  0.104  0.000 0.000      4.38      4.81
 6   548  0.148  0.145  0.000 0.000      5.54      5.56
 7   548  0.145  0.208  0.000 0.000      7.17      7.27
 8   548  0.106  0.208  0.068 0.000      7.66      6.05
 9   548  0.104  0.208  0.109 0.000      7.65      5.59
10   548  0.140  0.208  0.151 0.000      7.38      6.12
11   548  0.102  0.208  0.192 0.000      7.75      6.03
12   548  0.134  0.208  0.000 0.037      4.19      4.13
13   548  0.156  0.208  0.000 0.097      2.87      2.77
14   548  0.130  0.208  0.000 0.123      2.70      2.67
15   548  0.134  0.208  0.177 0.123      2.68      2.67
16   548  0.132  0.208  0.000 0.173      2.35      2.55
17   548  0.146  0.208  0.000 0.191      2.20      2.28
18   548  0.159  0.208  0.000 0.208      2.10      2.29
19   548  0.198  0.208  0.000 0.000      6.84      7.29
20   548  0.223  0.208  0.000 0.000      6.42      7.09
21   573  0.150  0.208  0.000 0.000     13.87     15.44
RMS error of ln(TOF): 0.131
```

Partial pressures are in atm, and TOFs are CO₂ per Pt site per minute.
- **Agreement:** the model reproduces the inhibition by H₂ (conditions 12–18)
  and the temperature dependence.
- **Discrepancy:** it overestimates the rates with CO₂ in the feed (conditions
  8–11) by 20–40%.

The same model in CatMap's conventions (`tests/catmap_wgs.py`) is used to
validate micki against CatMap ([CatMap conventions](../catmap.md)).
