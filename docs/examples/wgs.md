# Example: water-gas shift on Pt

The water-gas shift reaction, CO + H2O → CO2 + H2, on Pt(111), modeled after
Hermes et al., J. Chem. Phys. 151, 014112 (2019) and compared with the
experiments of Grabow et al. (J. Phys. Chem. C 112, 4608 (2008), Table 5) at 21
reaction conditions. The complete script is
[`tests/wgs.py`](../../tests/wgs.py), and it doubles as micki's regression
test. Running it regenerates `tests/data/wgs_reference.json`.

## Species

DFT energies (PBE-D3) and frequencies are in an ASE database,
`tests/data/wgs.json`. Energies are referenced to the clean slab, CO, H2O and
H2:

```python
sp = read_from_db('tests/data/wgs.json',
                  eref=['slab', 'co_g', 'h2o_g', 'h2_g'])
# The O-H-OH transition state (H transfer between two O) is end-to-end
# symmetric.
sp['o-h-oh'].symm = 2
```

The database row of each transition state lists its two sites, so the lattice
below gives them configurational entropy. Following the paper, the CO and H
binding energies are shifted to match experimental binding enthalpies. The OH
and COOH shifts are fitted to the experimental turnover frequencies (least
squares in ln TOF over all 21 conditions):

```python
sp['co'].dE = 0.09496182099234107
sp['h'].dE = 0.236689058 / 2.0
sp['oh'].dE = -0.2280
sp['cooh'].dE = 0.1603
```

Lateral interactions are linear in the coverages: CO with every adsorbate, and
O with itself, as symmetric pairs. For example:

```python
sp['co'].lateral = 2 * 0.784423808 * sp['co'].symbol
sp['o'].lateral = (1.147079243 * sp['co'].symbol
                   + 2 * 1.11913902 * sp['o'].symbol)
sp['co'].lateral += 1.147079243 * sp['o'].symbol
```

## Reactions

```python
{
    'co_ads': Reaction(sp['co_g'], sp['co'], method='STICK'),
    'h2o_ads': Reaction(sp['h2o_g'], sp['h2o'], method='STICK'),
    'h2_ads': Reaction(sp['h2_g'], 2 * sp['h'], method='STICK'),
    'ho-h': Reaction(sp['h2o'], sp['oh'] + sp['h'], ts=sp['ho-h']),
    'o-h': Reaction(sp['oh'], sp['o'] + sp['h'], ts=sp['o-h']),
    'o-h-oh': Reaction(2 * sp['oh'], sp['o'] + sp['h2o'], method='DIEQUIL'),
    'o-co': Reaction(sp['o'] + sp['co'], sp['co2_g'], ts=sp['o-co']),
    'co-oh': Reaction(sp['cooh'], sp['co'] + sp['oh'], ts=sp['co-oh']),
    'oco-h': Reaction(sp['cooh'] + sp['slab'], sp['co2_g'] + sp['h'],
                      ts=sp['oco-h']),
    'oco-h-o': Reaction(sp['cooh'] + sp['o'], sp['co2_g'] + sp['oh'],
                        method='EQUIL'),
    'oco-h-oh': Reaction(sp['cooh'] + sp['oh'], sp['co2_g'] + sp['h2o'],
                         method='EQUIL'),
}
```

The steps follow the paper. Adsorption is non-activated (collision theory).
Surface steps with a transition state follow their reactants' and products'
lateral interactions and energy shifts through the self-consistent α. The
remaining steps have no barrier. (With `DIEQUIL`, the `o-h-oh` transition
state isn't used.)

## Reactor model

Each condition is solved in two steps, as in `run_condition` in the script:

1. A CSTR at the inlet composition (fixed gases) is solved to steady state. It
   gives the surface coverages at the reactor inlet.
2. A PFR started from that state is integrated over the reactor's residence
   time. The adsorbates are at pseudo-steady state (algebraic), and the gases
   change along the reactor.

```python
model = Model(T, ASITE, reactor='CSTR')
model.lattice = {sp['slab']: {sp['slab']: 6}}      # hexagonal Pt(111)
model.add_reactions(rxns)
model.set_fixed(['co_g', 'h2o_g', 'h2_g', 'co2_g'])
model.set_initial_conditions(U0)                   # gases in M, coverages
_, U_cstr, _ = model.find_steady_state(maxiter=200000, epsilon=1e-8)

pfr = Model(T, ASITE, reactor='PFR', rhocat=1.)
pfr.lattice = model.lattice
pfr.add_reactions(rxns)
pfr.set_initial_conditions(U_cstr)
U_pfr, _ = pfr.solve(dt, 1000)                     # dt: residence time
```

The turnover frequency follows from the CO2 at the outlet, the flow rate and
the number of catalytic sites (`NSITES`, from the Pt loading and dispersion).

## Results

TOF in CO2 per site per minute (micki) against experiment:

| # | T (K) | p(CO) | p(H2O) | p(CO2) | p(H2) (atm) | micki | experiment |
|---|---|---|---|---|---|---|---|
| 1 | 523 | 0.154 | 0.208 | 0.000 | 0.000 | 3.36 | 3.68 |
| 2 | 548 | 0.055 | 0.208 | 0.000 | 0.000 | 8.08 | 8.56 |
| 3 | 548 | 0.105 | 0.208 | 0.000 | 0.000 | 7.46 | 8.06 |
| 4 | 548 | 0.137 | 0.062 | 0.000 | 0.000 | 2.98 | 3.63 |
| 5 | 548 | 0.144 | 0.104 | 0.000 | 0.000 | 4.38 | 4.81 |
| 6 | 548 | 0.148 | 0.145 | 0.000 | 0.000 | 5.54 | 5.56 |
| 7 | 548 | 0.145 | 0.208 | 0.000 | 0.000 | 7.17 | 7.27 |
| 8 | 548 | 0.106 | 0.208 | 0.068 | 0.000 | 7.66 | 6.05 |
| 9 | 548 | 0.104 | 0.208 | 0.109 | 0.000 | 7.65 | 5.59 |
| 10 | 548 | 0.140 | 0.208 | 0.151 | 0.000 | 7.38 | 6.12 |
| 11 | 548 | 0.102 | 0.208 | 0.192 | 0.000 | 7.75 | 6.03 |
| 12 | 548 | 0.134 | 0.208 | 0.000 | 0.037 | 4.19 | 4.13 |
| 13 | 548 | 0.156 | 0.208 | 0.000 | 0.097 | 2.87 | 2.77 |
| 14 | 548 | 0.130 | 0.208 | 0.000 | 0.123 | 2.70 | 2.67 |
| 15 | 548 | 0.134 | 0.208 | 0.177 | 0.123 | 2.68 | 2.67 |
| 16 | 548 | 0.132 | 0.208 | 0.000 | 0.173 | 2.35 | 2.55 |
| 17 | 548 | 0.146 | 0.208 | 0.000 | 0.191 | 2.20 | 2.28 |
| 18 | 548 | 0.159 | 0.208 | 0.000 | 0.208 | 2.10 | 2.29 |
| 19 | 548 | 0.198 | 0.208 | 0.000 | 0.000 | 6.84 | 7.29 |
| 20 | 548 | 0.223 | 0.208 | 0.000 | 0.000 | 6.42 | 7.09 |
| 21 | 573 | 0.150 | 0.208 | 0.000 | 0.000 | 13.87 | 15.44 |

The root-mean-square error of ln TOF is 0.131. The model reproduces the
inhibition by H2 (conditions 12–18) and the temperature dependence. It
overestimates the rates with CO2 in the feed (conditions 8–11) by 20–40%.

The same model in CatMap's conventions (`tests/catmap_wgs.py`) is used to
validate micki against CatMap ([CatMap conventions](../catmap.md)).
