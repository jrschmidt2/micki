"""The water-gas shift model in CatMap's conventions, for test_catmap.py.

micki's options for CatMap's conventions applied to the WGS model of wgs.py:
gases referenced to 1 bar (Gas pref=1), every barrier clipped at the higher
of the initial and final states at the current coverages
(clip='coverage'), CO adsorption non-activated (collision theory, STICK;
CatMap computes it with standard atomic weights, so CO gets those masses
here), the other steps without a transition state barrierless (EQUIL), no
configurational entropy (no lattice), fixed gas pressures. Lateral
interactions are CatMap's first-order model with the WGS interaction
parameters, in two variants:

- 'intermediate': linear response; transition states follow the initial
  and final states with weight 1/2 each (alpha=0.5; CatMap's
  'intermediate_state').
- 'piecewise': piecewise-linear response (cutoff 0.25); transition states
  get explicit interaction rows equal to the sum of the initial-state rows
  (explicit_ts=True; CatMap's 'initial_state').

Running this file (micki environment) writes the model in CatMap's terms
to data/catmap_wgs_input.json; catmap_reference.py (CatMap environment)
solves it with CatMap and writes data/catmap_wgs_reference.json.
"""

import json
import os
import sys

import sympy as sym
from ase.data import atomic_masses, atomic_numbers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wgs  # noqa: E402
from micki import Model, Reaction  # noqa: E402
from micki.lateral import first_order  # noqa: E402
from micki.model import _at_zero_coverage  # noqa: E402
from micki.utils import bar_to_molar  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
INPUT = os.path.join(DATA_DIR, 'catmap_wgs_input.json')
REFERENCE = os.path.join(DATA_DIR, 'catmap_wgs_reference.json')

VARIANTS = ('intermediate', 'piecewise')
CONDITIONS = (1, 7, 13, 21)  # of wgs.CONDITIONS, counted from 1
GASES = {'co_g': 'CO_g', 'h2o_g': 'H2O_g', 'h2_g': 'H2_g', 'co2_g': 'CO2_g'}
ADS = {'co': 'CO_s', 'h2o': 'H2O_s', 'oh': 'OH_s', 'o': 'O_s', 'h': 'H_s',
       'cooh': 'COOH_s'}
# reaction: (initial-state adsorbates, final-state adsorbates, CatMap TS
# name, CatMap expression); TS names encode their composition
STEPS = {
    'co_ads': ([], ['co'], None, 'CO_g + *_s -> CO_s'),
    'h2o_ads': ([], ['h2o'], None, 'H2O_g + *_s -> H2O_s'),
    'h2_ads': ([], ['h', 'h'], None, 'H2_g + 2*_s -> 2H_s'),
    'ho-h': (['h2o'], ['oh', 'h'], 'HO-H_s',
             'H2O_s + *_s <-> HO-H_s + *_s -> OH_s + H_s'),
    'o-h': (['oh'], ['o', 'h'], 'O-H_s',
            'OH_s + *_s <-> O-H_s + *_s -> O_s + H_s'),
    'o-h-oh': (['oh', 'oh'], ['o', 'h2o'], None, '2OH_s -> O_s + H2O_s'),
    'o-co': (['o', 'co'], [], 'O-CO_s',
             'O_s + CO_s <-> O-CO_s + *_s -> CO2_g + 2*_s'),
    'co-oh': (['cooh'], ['co', 'oh'], 'CO-OH_s',
              'COOH_s + *_s <-> CO-OH_s + *_s -> CO_s + OH_s'),
    'oco-h': (['cooh'], ['h'], 'OCO-H_s',
              'COOH_s + *_s <-> OCO-H_s + *_s -> CO2_g + H_s + *_s'),
    'oco-h-o': (['cooh', 'o'], ['oh'], None,
                'COOH_s + O_s -> CO2_g + OH_s + *_s'),
    'oco-h-oh': (['cooh', 'oh'], ['h2o'], None,
                 'COOH_s + OH_s -> CO2_g + H2O_s + *_s'),
}
CO2_STEPS = ('o-co', 'oco-h', 'oco-h-o', 'oco-h-oh')


def build(variant):
    """Species, reactions and interaction rows of the CatMap-style model."""
    sp = wgs.build_species()
    for name in GASES:
        sp[name].pref = 1.
    co_g = sp['co_g']
    co_g.mass = [atomic_masses[atomic_numbers[a.symbol]] for a in co_g.atoms]
    ads = [sp[name] for name in ADS]
    # interaction rows from the WGS lateral interactions (linear in the
    # coverages)
    eps = {}
    for i in ads:
        terms = sym.sympify(i.lateral).as_coefficients_dict()
        eps[i] = {j: float(terms[j.symbol]) for j in ads
                  if terms.get(j.symbol)}
    rows = dict(eps)
    if variant == 'piecewise':
        for name, (initial, _, ts, _) in STEPS.items():
            if ts is None:
                continue
            row = {}
            for i in initial:
                for j, value in eps[sp[i]].items():
                    row[j] = row.get(j, 0.) + value
            rows[sp[name]] = row
    response = 'linear' if variant == 'intermediate' else 'piecewise_linear'
    first_order(ads, rows, response=response, cutoff=0.25)

    ts_kw = ({'alpha': 0.5} if variant == 'intermediate'
             else {'explicit_ts': True})
    clip = {'clip': 'coverage'}

    def tst(react, prod, ts):
        return Reaction(react, prod, ts=sp[ts], **ts_kw, **clip)

    rxns = {
        'co_ads': Reaction(sp['co_g'], sp['co'], method='STICK', **clip),
        'h2o_ads': Reaction(sp['h2o_g'], sp['h2o'], method='EQUIL', **clip),
        'h2_ads': Reaction(sp['h2_g'], 2 * sp['h'], method='EQUIL', **clip),
        'ho-h': tst(sp['h2o'], sp['oh'] + sp['h'], 'ho-h'),
        'o-h': tst(sp['oh'], sp['o'] + sp['h'], 'o-h'),
        'o-h-oh': Reaction(2 * sp['oh'], sp['o'] + sp['h2o'], method='EQUIL',
                           **clip),
        'o-co': tst(sp['o'] + sp['co'], sp['co2_g'], 'o-co'),
        'co-oh': tst(sp['cooh'], sp['co'] + sp['oh'], 'co-oh'),
        'oco-h': tst(sp['cooh'] + sp['slab'], sp['co2_g'] + sp['h'], 'oco-h'),
        'oco-h-o': Reaction(sp['cooh'] + sp['o'], sp['co2_g'] + sp['oh'],
                            method='EQUIL', **clip),
        'oco-h-oh': Reaction(sp['cooh'] + sp['oh'], sp['co2_g'] + sp['h2o'],
                             method='EQUIL', **clip),
    }
    return sp, rxns, eps, response


def pressures(condition):
    """Temperature and partial pressures (bar) of condition number
    `condition` (from 1) of wgs.CONDITIONS."""
    T, p_co, p_h2o, p_co2, p_h2, flow = wgs.CONDITIONS[condition - 1]
    atm = 1.01325
    return T, {'co_g': atm * p_co, 'h2o_g': atm * p_h2o,
               'co2_g': atm * p_co2, 'h2_g': atm * p_h2}


def solve(variant, condition, analytic_jac=False):
    """micki's steady state: coverages and CO2 formation rate (1/s)."""
    sp, rxns, _, _ = build(variant)
    T, p = pressures(condition)
    model = Model(T, wgs.ASITE, reactor='CSTR', analytic_jac=analytic_jac)
    model.add_reactions(rxns)
    model.set_fixed(list(GASES))
    U0 = {g: bar_to_molar(p[g], T) for g in GASES}
    U0.update(wgs.COVERAGE0)
    model.set_initial_conditions(U0)
    _, U, r = model.find_steady_state()
    return ({name: float(U[name]) for name in ADS},
            sum(float(r[name]) for name in CO2_STEPS))


def export(variant, condition):
    """The model in CatMap's terms: free energies at zero coverage
    (adsorbates and transition states relative to bare sites, gases at
    1 bar), interaction rows, pressures."""
    sp, rxns, eps, response = build(variant)
    T, p = pressures(condition)
    for rxn in rxns.values():
        rxn.update(T=T, Asite=wgs.ASITE, L=0)
    G_slab = _at_zero_coverage(sp['slab'].get_G(T))

    def G(species):
        if species is sp['slab']:
            return 0.
        value = _at_zero_coverage(species.get_G(T))
        return value - len(species.sites) * G_slab if species.sites \
            else value

    energies = {GASES[g]: G(sp[g]) for g in GASES}
    energies.update({ADS[a]: G(sp[a]) for a in ADS})
    rows = {ADS[i.label]: {ADS[j.label]: v for j, v in row.items()}
            for i, row in eps.items()}
    steps = {}
    for name, (initial, final, ts, expression) in STEPS.items():
        step = {'expression': expression, 'ts': ts}
        if ts is not None:
            rxn = rxns[name]
            G_initial = sum(G(s) for s in rxn.reactants)
            step['E_ts'] = G_initial + _at_zero_coverage(rxn.dG_act)
            row = {}
            weights = ([(initial, 0.5), (final, 0.5)]
                       if variant == 'intermediate' else [(initial, 1.)])
            for names, w in weights:
                for i in names:
                    for j, value in rows[ADS[i]].items():
                        row[j] = row.get(j, 0.) + w * value
            step['eps'] = row
        if name == 'co_ads':
            # collision theory per bar, site area in A^2
            step['prefactor'] = {'type': 'non-activated',
                                 'A_site': wgs.ASITE * 1e20}
        steps[name] = step
    return {'variant': variant, 'condition': condition, 'T': T,
            'response': response, 'cutoff': 0.25,
            'pressures': {GASES[g]: p[g] for g in GASES},
            'energies': energies, 'eps': rows, 'steps': steps}


if __name__ == '__main__':
    data = [export(v, c) for v in VARIANTS for c in CONDITIONS]
    with open(INPUT, 'w') as f:
        json.dump(data, f, indent=1)
        f.write('\n')
    print('Wrote', INPUT)
