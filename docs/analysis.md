# Sensitivity analysis

`micki.ModelAnalysis` computes steady-state sensitivities by finite
differences: each quantity is a central difference between two steady states.
Definitions and discussion are in Hermes et al., J. Chem. Phys. 151, 014112
(2019) and Hermes' dissertation (ch. 4).

```python
ModelAnalysis(model, product_reaction, Uequil)
```

`product_reaction` names the reaction whose net rate r is analyzed. Pick a
step that is not nearly equilibrated, e.g. the product-forming step: the net
rate of an equilibrated step is a small difference of large fluxes, and too
imprecise for finite differences. `Uequil` are the initial conditions, as for
`Model.set_initial_conditions`. The reference steady state is found from them
when the analysis is created (`analysis.U`, `analysis.r`).

| Method | Quantity |
|---|---|
| `campbell_rate_control(rxn_name, scale=0.001)` | Degree of rate control X = (k/r) ∂r/∂k, with the forward and reverse rate constants of `rxn_name` scaled together (keq fixed). With fixed fluids, the X of all steps sum to 1. |
| `thermodynamic_rate_control(names, dg=None)` | Degree of thermodynamic rate control −(kT/r) ∂r/∂G of the species `names` (a label or a list, shifted together; default step 0.001 kT). |
| `activation_barrier(dT=0.01)` | Apparent activation energy kT² ∂ln r/∂T (eV). |
| `rate_order(name, drho=0.05)` | Reaction order ∂ln r/∂ln c in the (fixed) species `name`. |
| `drate_order_dg(fluid, adsorbates, rho_scale=0.01, g_scale=0.01)` | Change of the rate order in `fluid` with the free energy of `adsorbates`. |

Thermodynamic rate control shifts the free energies through the species'
`dE`. With micki's default conventions, the attached transition states follow
the shift through α, as the dissertation defines it (eq. 4.4). Under
`micki.conventions('catmap')` they stay fixed, as in Campbell's original
definition.

## Example

For the CO oxidation model of the [user guide](user-guide.md#a-complete-example):

```python
from micki import ModelAnalysis

U0 = {'co_g': bar_to_molar(0.1, T), 'o2_g': bar_to_molar(0.2, T),
      'co2_g': 0.}
analysis = ModelAnalysis(model, 'co_ox', U0)
for name in reactions:
    print('DRC %-7s %+.3f' % (name, analysis.campbell_rate_control(name)))
for name in ['co', 'o']:
    print('TRC %-7s %+.3f' % (name, analysis.thermodynamic_rate_control(name)))
print('apparent barrier %.3f eV' % analysis.activation_barrier())
print('order in CO %+.3f, in O2 %+.3f' % (analysis.rate_order('co_g'),
                                          analysis.rate_order('o2_g')))
```

```
DRC co_ads  +1.944
DRC o2_ads  -0.993
DRC co_ox   +0.049
TRC co      +0.028
TRC o       -0.096
apparent barrier 0.649 eV
order in CO +1.987, in O2 -0.996
```

The surface is nearly covered by O, and CO adsorption onto the few free sites
controls the rate. The free sites are set by the balance of O₂ adsorption
(on pairs of sites) and O removal, so they scale as p(CO)/p(O₂), and the rate
as p(CO)²/p(O₂): orders +2 and −1. Faster O₂ adsorption would only poison the
surface further, hence its negative degree of rate control. The degrees of
rate control sum to 1.000.

## Accuracy

Each value is a central difference of two steady states, so its error is
about (relative error of the steady-state rate) / (relative step).
- **Steady-state precision:** steady states found by Newton are converged to
  about 1e-10 relative.
- **Default steps:** these give 3–5 significant digits. A larger step (e.g.
  `scale=0.01`) is more robust when the rate is noisy, at the cost of a
  second-order truncation error.
