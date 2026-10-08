"""Two made-up microkinetic models in CatMap's input terms, for
test_catmap.py: ammonia synthesis on a Ru-like surface (N2 dissociation and
three hydrogenation transition states, non-activated H2 adsorption, linear
first-order interactions) and ethylene hydrogenation with H on its own site
type (interactions within and across sites, smooth piecewise-linear
response). Energies are formation energies in eV as in CatMap input tables,
frequencies in cm^-1 (gases: CatMap's experimental frequencies; transition
states: real modes only).

CatMap solves them natively (catmap_models_reference.py, CatMap
environment: ideal_gas and harmonic_adsorbate thermochemistry, CatMap's
geometric-mean cross terms and intermediate_state TS weighting);
build() makes the same models in micki under micki.conventions('catmap'),
where micki computes all thermochemistry itself and only takes the
adsorbate-adsorbate interaction matrix from CatMap. This module imports
micki only inside build(), so the CatMap environment can use the specs.
"""

import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
REFERENCE = os.path.join(DATA_DIR, 'catmap_models_reference.json')

CM = 1.239842e-4  # CatMap's cm^-1 -> eV

# CatMap's ideal_gas_params: symmetry number, geometry, spin
GAS_PARAMS = {
    'H2_g': (2, 'linear', 0), 'N2_g': (2, 'linear', 0),
    'NH3_g': (3, 'nonlinear', 0), 'C2H4_g': (4, 'nonlinear', 0),
    'C2H6_g': (6, 'nonlinear', 0),
}

# CatMap's experimental_gas_frequencies (cm^-1)
GAS_FREQS = {
    'H2_g': [4401], 'N2_g': [2359],
    'NH3_g': [3337, 950, 3444, 3444, 1627, 1627],
    'C2H4_g': [3026, 1623, 1342, 1023, 3103, 1236, 949, 943, 3106, 826,
               2989, 1444],
    'C2H6_g': [2954, 1388, 995, 289, 2896, 1379, 2969, 2969, 1468, 1468,
               1190, 1190, 2985, 2985, 1469, 1469, 822, 822],
}

# formation energies relative to N2 and H2
NH3 = {
    'name': 'NH3 synthesis (Ru-like)',
    'sites': ['s'],
    'gases': {'N2_g': 0., 'H2_g': 0., 'NH3_g': -0.47},
    'adsorbates': {
        'N_s': (-0.40, [560, 470, 470]),
        'H_s': (-0.30, [1050, 800, 800]),
        'NH_s': (-0.20, [3350, 1050, 700, 700, 450, 450]),
        'NH2_s': (-0.35, [3400, 3300, 1500, 700, 650, 600, 450, 400, 300]),
        'NH3_s': (-1.05, [3450, 3450, 3350, 1600, 1600, 1150, 650, 650,
                          400, 300, 150, 150]),
    },
    'ts': {
        'N-N_s': (1.00, [800, 450, 400, 300, 200]),
        'N-H_s': (0.45, [900, 600, 550, 450, 350]),
        'NH-H_s': (0.35, [3350, 1000, 700, 600, 500, 400, 350, 300]),
        'NH2-H_s': (-0.05, [3400, 3300, 1450, 1000, 700, 600, 500, 400, 350,
                            300, 200]),
    },
    'reactions': [
        ('N2_g + 2*_s <-> N-N_s + *_s -> 2N_s', None),
        ('H2_g + 2*_s -> 2H_s', 'non-activated'),
        ('N_s + H_s <-> N-H_s + *_s -> NH_s + *_s', None),
        ('NH_s + H_s <-> NH-H_s + *_s -> NH2_s + *_s', None),
        ('NH2_s + H_s <-> NH2-H_s + *_s -> NH3_s + *_s', None),
        ('NH3_s -> NH3_g + *_s', None),
    ],
    'conditions': [
        (600., {'N2_g': 2.5, 'H2_g': 7.5, 'NH3_g': 0.1}),
        (700., {'N2_g': 0.5, 'H2_g': 1.5, 'NH3_g': 0.05}),
        (650., {'N2_g': 25., 'H2_g': 75., 'NH3_g': 1.}),
    ],
    'A_site': 6.4,  # A^2
    'interactions': {
        'response': 'linear',
        'params': {'slope': 1, 'cutoff': 0.25, 'smoothing': 0.05},
        'self': {'N_s': 1.5, 'H_s': 0.2, 'NH_s': 1.0, 'NH2_s': 0.8,
                 'NH3_s': 0.5},
        'cross_mode': 'geometric_mean',
    },
}

# Horiuti-Polanyi with H on its own site type; formation energies relative
# to C2H4 and H2
C2H4 = {
    'name': 'C2H4 hydrogenation (two sites)',
    'sites': ['s', 'h'],
    'gases': {'C2H4_g': 0., 'H2_g': 0., 'C2H6_g': -1.40},
    'adsorbates': {
        'C2H4_s': (-0.90, [3050, 3000, 2950, 2900, 1450, 1400, 1200, 1100,
                           950, 900, 800, 700, 450, 350, 300, 250, 150,
                           100]),
        'C2H5_s': (-1.00, [3000, 2950, 2900, 2850, 2800, 1450, 1430, 1400,
                           1350, 1150, 1100, 1000, 950, 850, 750, 500, 400,
                           300, 250, 200, 100]),
        'H_h': (-0.35, [1100, 750, 750]),
    },
    'ts': {
        'C2H4-H_s': (-0.40, [3050, 3000, 2950, 2900, 1450, 1400, 1300, 1150,
                             1050, 950, 900, 800, 700, 600, 450, 350, 300,
                             250, 150, 100]),
        'C2H5-H_s': (-0.75, [3000, 2950, 2900, 2850, 2800, 1450, 1430, 1400,
                             1350, 1300, 1150, 1100, 1000, 950, 850, 750, 600,
                             500, 400, 300, 250, 200, 100]),
    },
    'reactions': [
        ('H2_g + 2*_h -> 2H_h', 'non-activated'),
        ('C2H4_g + *_s -> C2H4_s', None),
        ('C2H4_s + H_h <-> C2H4-H_s + *_h -> C2H5_s + *_h', None),
        ('C2H5_s + H_h <-> C2H5-H_s + *_h -> C2H6_g + *_s + *_h', None),
    ],
    'conditions': [
        (300., {'C2H4_g': 0.1, 'H2_g': 0.2, 'C2H6_g': 0.01}),
        (350., {'C2H4_g': 0.25, 'H2_g': 0.25, 'C2H6_g': 0.}),
        (400., {'C2H4_g': 0.05, 'H2_g': 1., 'C2H6_g': 0.1}),
    ],
    'A_site': 6.4,
    'interactions': {
        'response': 'smooth_piecewise_linear',
        'params': {'slope': 1, 'cutoff': 0.25, 'smoothing': 0.05},
        'self': {'C2H4_s': 2.0, 'C2H5_s': 1.5, 'H_h': 0.3},
        'cross_mode': 'geometric_mean',
    },
}

MODELS = {'nh3': NH3, 'c2h4': C2H4}


def parse_side(side):
    """'2*_s + O2_g' -> [('*_s', 2), ('O2_g', 1)]"""
    terms = []
    for term in side.split('+'):
        term = term.strip()
        n = ''
        while term[0].isdigit():
            n += term[0]
            term = term[1:]
        terms.append((term, int(n) if n else 1))
    return terms


def parse_reaction(expression):
    """Initial state, transition state (or None), final state."""
    states = expression.replace('<->', '|').replace('->', '|').split('|')
    if len(states) == 2:
        return parse_side(states[0]), None, parse_side(states[1])
    return parse_side(states[0]), parse_side(states[1]), \
        parse_side(states[2])


def build(m, catmap_matrix):
    """The model in micki under CatMap's conventions: species, vacancies,
    reactions ('r0', 'r1', ... in the order of m['reactions'])."""
    import numpy as np
    from ase import Atoms
    from ase.build import molecule
    from ase.calculators.singlepoint import SinglePointCalculator

    import micki
    from micki import Adsorbate, Gas, Reaction
    from micki.lateral import first_order

    def with_energy(atoms, E):
        atoms.calc = SinglePointCalculator(atoms, energy=E)
        return atoms

    sp = {}
    with micki.conventions('catmap'):
        sites = {s: Adsorbate(with_energy(Atoms(), 0.), s, np.array([]))
                 for s in m['sites']}
        for g, E in m['gases'].items():
            symm, geometry, spin = GAS_PARAMS[g]
            atoms = molecule(g[:-2])  # CatMap's ideal_gas geometries
            vib = [f * CM for f in GAS_FREQS[g]]
            ncut = 3 * len(atoms) - len(vib)
            sp[g] = Gas(with_energy(atoms, E), g, np.array([0.] * ncut + vib),
                        symm=symm, spin=spin)
            if sp[g].linear != (geometry == 'linear'):
                raise ValueError('geometry of {}'.format(g))
        for name, (E, freqs) in m['adsorbates'].items():
            base, site = name.rsplit('_', 1)
            sp[name] = Adsorbate(with_energy(Atoms(base), E), name,
                                 np.array([f * CM for f in freqs]),
                                 sites=[sites[site]])
        for name, (E, freqs) in m['ts'].items():
            base, site = name.rsplit('_', 1)
            # the first (imaginary) mode is dropped for transition states
            sp[name] = Adsorbate(with_energy(Atoms(base.replace('-', '')), E),
                                 name,
                                 np.array([-0.01] + [f * CM for f in freqs]),
                                 ts=True, sites=[sites[site]])

        # CatMap's reaction expressions, read with its species names
        names = dict(sp)
        names.update({'*_' + site: vac for site, vac in sites.items()})
        rxns = {}
        for i, (expression, pf) in enumerate(m['reactions']):
            kw = {'method': 'STICK'} if pf == 'non-activated' else {}
            rxns['r%d' % i] = Reaction.from_string(expression, names, **kw)

        ads = [sp[n] for n in m['adsorbates']]
        eps = {sp[a]: {sp[b]: v for b, v in catmap_matrix[a].items()
                       if b in m['adsorbates'] and v}
               for a in m['adsorbates']}
        params = m['interactions']['params']
        first_order(ads, eps, response=m['interactions']['response'],
                    slope=params['slope'], cutoff=params['cutoff'],
                    smoothing=params['smoothing'])
    return sp, sites, rxns


def forward_fluxes(m, c):
    """Forward fluxes kf * prod(p) * prod(theta) of a CatMap solution."""
    theta = dict(c['coverage'])
    for site in m['sites']:
        theta['*_' + site] = 1. - sum(v for a, v in c['coverage'].items()
                                      if a.endswith('_' + site))
    out = []
    for i, (expression, _) in enumerate(m['reactions']):
        initial, _, _ = parse_reaction(expression)
        f = c['kf'][i]
        for name, n in initial:
            f *= (c['p'][name] if name.endswith('_g') else theta[name]) ** n
        out.append(f)
    return out
