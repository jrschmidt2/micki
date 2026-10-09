"""Two electrochemical models in CatMap's input terms, for test_catmap.py:
oxygen reduction (2e- and 4e- pathways, transport through a "double
layer" site, PCET steps with barriers at the equilibrium potential and
beta = 0.5 or 0.1, thermal O-O scission) and hydrogen evolution
(Volmer-Heyrovsky), with Pt(111) free energies from Hansen et al., J. Phys.
Chem. C 118, 6706 (2014). They are J. R. Schmidt's CatMap templates
(ORR.mkm, ORR_input.txt), solved by CatMap at a series of potentials
(catmap_echem_reference.py, CatMap environment).

CatMap's proton-electron pair pe_g at pressure 1 corresponds in micki to
H3O+ at 1 M (pH 0, where the SHE and RHE scales coincide) plus an
electron, each releasing a water molecule at activity 1. build() makes
the models in micki that way; this module imports micki only inside
build(), so the CatMap environment can use the specs.
"""

import os
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
REFERENCE = os.path.join(DATA_DIR, 'catmap_echem_reference.json')

T = 298.

# free energies (eV) relative to H2(g) and H2O(l), CatMap's frozen
# formation energies: name -> (site, energy); site 'gas' for gases
ENERGIES = {
    'pe_g': ('gas', 0.), 'H2_g': ('gas', 0.), 'H2O_g': ('gas', 0.),
    'O2_g': ('gas', 4.739), 'H2O2_g': ('gas', 3.113),
    'H2O2_a': ('a', 3.363), 'H2O2_dl': ('dl', 3.113), 'O2_a': ('a', 4.563),
    'O_a': ('a', 1.704), 'OH_a': ('a', 1.067), 'H_a': ('a', 0.3),
    'OOH_a': ('a', 4.029), 'O2_dl': ('dl', 4.739),
    'OH-O_a': ('a', 4.639), 'O-O_a': ('a', 5.203),
}

ORR = {
    'name': 'ORR (Pt, Hansen 2014)',
    'sites': ['a', 'dl'],
    # expression, prefactor (None: kT/h), beta
    'reactions': [
        ('O2_g + *_dl -> O2_dl', 8e5, 0.5),
        ('O2_dl + *_a -> O2_a + *_dl', 1e8, 0.5),
        ('O2_a + pe_g <-> ^0.26eV_a -> OOH_a', 1e9, 0.5),
        ('OOH_a + pe_g <-> ^1.55eV_a -> O_a + H2O_g', 1e9, 0.5),
        ('O_a + pe_g <-> ^0.26eV_a -> OH_a', 1e9, 0.5),
        ('OH_a + pe_g <-> ^0.26eV_a -> H2O_g + *_a', 1e9, 0.5),
        ('OOH_a + pe_g <-> ^0.26eV_a -> H2O2_a', 1e9, 0.5),
        ('H2O2_a + pe_g <-> ^0.38eV_a -> H2O_g + OH_a', 1e9, 0.1),
        ('H2O2_a + *_dl -> H2O2_dl + *_a', 1e8, 0.5),
        ('H2O2_dl -> H2O2_g + *_dl', 8e5, 0.5),
        ('OOH_a + *_a <-> OH-O_a + *_a -> OH_a + O_a', None, 0.5),
        ('O2_a + *_a <-> O-O_a + *_a -> O_a + O_a', None, 0.5),
    ],
    'pressures': {'H2O_g': 0., 'H2O2_g': 0., 'O2_g': 2.34e-5, 'pe_g': 1.},
    'voltages': [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0],
}

HER = {
    'name': 'HER (Pt, Hansen 2014)',
    'sites': ['a'],
    'reactions': [
        ('*_a + pe_g <-> ^0.26eV_a -> H_a', 1e9, 0.5),
        ('H_a + pe_g <-> ^0.26eV_a -> H2_g + *_a', 1e9, 0.5),
    ],
    'pressures': {'H2_g': 1., 'pe_g': 1.},
    'voltages': [-0.5, -0.3, -0.1, 0.1],
}

MODELS = {'orr': ORR, 'her': HER}


def species_names(m):
    """Species (not sites) of a model, in CatMap's names."""
    names = set()
    for expression, _, _ in m['reactions']:
        for name in re.split(r'<->|->|\+', expression):
            name = name.strip().lstrip('0123456789')
            if name and not name.startswith('*') and not name.startswith('^'):
                names.add(name)
    return sorted(names)


def micki_expression(expression):
    """CatMap expression -> micki expression: pe_g becomes h3o_aq + e, and
    each releases a water molecule (h2o_l, activity 1)."""
    states = [s.strip() for s in re.split(r'(<->|->)', expression)]
    n_pe = states[0].count('pe_g')
    states[0] = states[0].replace('pe_g', 'h3o_aq + e')
    if n_pe:
        states[-1] += ' + h2o_l' * n_pe
    return ' '.join(states)


def build(m):
    """The model in micki: species dict (micki labels), reactions ('r0',
    'r1', ... in the order of m['reactions'])."""
    from ase import Atoms
    import micki
    from micki import Adsorbate, Electron, Reaction, Solute

    def formula(name):
        return name.rsplit('_', 1)[0].replace('-', '')

    names = {}
    sites = {s: Adsorbate(Atoms(), 'site_' + s, E=0.) for s in m['sites']}
    for s, site in sites.items():
        names['*_' + s] = site
    for name in species_names(m):
        if name == 'pe_g':
            continue
        site, E = ENERGIES[name]
        if site == 'gas':
            # CatMap's frozen gases at pressure p: solutes with G = E and
            # activity p (no kT: the given value is the free energy)
            names[name] = Solute(name, E, formula(name))
        else:
            names[name] = Adsorbate(Atoms(formula(name)), name, E=E,
                                    ts='-' in name, sites=[sites[site]])
    names['h3o_aq'] = Solute('h3o_aq', 0., 'H3O', charge=1)
    names['h2o_l'] = Solute('h2o_l', 0., 'H2O')
    names['e'] = Electron('e')
    rxns = {}
    with micki.conventions('micki'):
        for i, (expression, prefactor, beta) in enumerate(m['reactions']):
            kw = {'clip': 'coverage', 'prefactor': prefactor}
            if 'pe_g' in expression:
                kw['beta'] = beta
            rxns['r%d' % i] = Reaction.from_string(
                micki_expression(expression), names, **kw)
    return names, rxns


def initial(m):
    """Fixed concentrations (activities) for micki: CatMap's pressures, with
    pe_g as H3O+ at 1 M and water (from the protons) at activity 1."""
    U0 = {name: p for name, p in m['pressures'].items() if name != 'pe_g'}
    U0['h3o_aq'] = m['pressures']['pe_g']
    U0['h2o_l'] = 1.
    return U0
