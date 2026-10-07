"""Solve the CatMap-style WGS model of catmap_wgs.py with CatMap.

Run in an environment with CatMap 0.3.1 (it needs numpy < 1.24, so not in
micki's environment), from the repository root, after
`python tests/catmap_wgs.py` (micki environment):

    python tests/catmap_reference.py

It writes CatMap input files to a temporary directory, replaces CatMap's
hard-coded CODATA-2010 kB and h with ASE's (which micki uses; otherwise
rates differ by ~5e-6), and integrates CatMap's own steady-state equations
(its generated d(theta)/dt, in mpmath) from an empty surface with SciPy's
Radau solver, since CatMap's Newton solvers do not converge for this
model. The coverages and the CO2 formation rate (CatMap's rates of the
CO2-forming steps) are written to data/catmap_wgs_reference.json.
"""

import contextlib
import io
import json
import os
import tempfile
import warnings

import mpmath
import numpy as np
from ase.units import kB as ASE_KB, _hplanck, _e
from scipy.integrate import solve_ivp

warnings.simplefilter('ignore')

from catmap import ReactionModel  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
INPUT = os.path.join(DATA_DIR, 'catmap_wgs_input.json')
REFERENCE = os.path.join(DATA_DIR, 'catmap_wgs_reference.json')
CO2_STEPS = ('o-co', 'oco-h', 'oco-h-o', 'oco-h-oh')
ASE_H = _hplanck / _e  # eV s


def _constant(name, value):
    # ReactionModel assigns its CODATA-2010 constants in __init__ and run();
    # a property on the class replaces every assignment with ASE's value
    attr = '_ase' + name

    def get(self):
        return self.__dict__[attr]

    def set(self, v):
        self.__dict__[attr] = (mpmath.mpf(repr(value))
                               if isinstance(v, mpmath.mpf) else value)
    return property(get, set)


ReactionModel._kB = _constant('_kB', ASE_KB)
ReactionModel._h = _constant('_h', ASE_H)


def write_inputs(d, workdir):
    names = list(d['eps'])
    for a in names:
        for b in names:
            if d['eps'][a].get(b, 0.) != d['eps'][b].get(a, 0.):
                raise ValueError('interaction matrix is not symmetric')
    rows = ['surface_name\tsite_name\tspecies_name\tformation_energy\t'
            'bulk_structure\tfrequencies\tother_parameters\treference']
    for name, E in d['energies'].items():
        if name.endswith('_g'):
            rows.append('None\tgas\t%s\t%.15f\tNone\t[]\t[]\tmicki'
                        % (name[:-2], E))
        else:
            rows.append('Pt\t111\t%s\t%.15f\tfcc\t[]\t[]\tmicki'
                        % (name[:-2], E))
    for step in d['steps'].values():
        if step['ts'] is not None:
            rows.append('Pt\t111\t%s\t%.15f\tfcc\t[]\t[]\tmicki'
                        % (step['ts'][:-2], step['E_ts']))
    with open(os.path.join(workdir, 'energies.txt'), 'w') as f:
        f.write('\n'.join(rows) + '\n')

    steps = list(d['steps'].values())
    lines = [
        "scaler = 'ThermodynamicScaler'",
        'rxn_expressions = %r' % [s['expression'] for s in steps],
        'prefactor_list = %r' % [s.get('prefactor') for s in steps],
        "surface_names = ['Pt']",
        "descriptor_names = ['temperature', 'logPressure']",
        'descriptor_ranges = [[%r, %r], [0, 0]]' % (d['T'], d['T']),
        'resolution = 1',
        'species_definitions = {}',
        "species_definitions['s'] = {'site_names': ['111'], 'total': 1}",
        "data_file = 'wgs.pkl'",
        "input_file = 'energies.txt'",
        "gas_thermo_mode = 'frozen_gas'",
        "adsorbate_thermo_mode = 'frozen_adsorbate'",
        'decimal_precision = 100',
        "adsorbate_interaction_model = 'first_order'",
        'interaction_response_function = %r' % d['response'],
        "interaction_response_parameters = {'slope': 1, 'cutoff': %r, "
        "'smoothing': 0}" % d['cutoff'],
        "cross_interaction_mode = 'neglect'",
        "transition_state_cross_interaction_mode = 'initial_state'",
    ]
    for name, p in d['pressures'].items():
        lines.append("species_definitions[%r] = {'concentration': %r}"
                     % (name, p))
    for i, a in enumerate(names):
        # every pair explicitly, including zeros: explicit parameters
        # override CatMap's own cross terms and transition-state weighting
        cross = {b: [d['eps'][a].get(b, 0.)] for b in names[i + 1:]}
        for step in steps:
            if step['ts'] is not None:
                cross[step['ts']] = [step['eps'].get(a, 0.)]
        lines.append("species_definitions[%r] = {'self_interaction_"
                     "parameter': [%r], 'cross_interaction_parameters': %r}"
                     % (a, d['eps'][a].get(a, 0.), cross))
    with open(os.path.join(workdir, 'wgs.mkm'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def solve(d):
    with tempfile.TemporaryDirectory() as workdir:
        write_inputs(d, workdir)
        cwd = os.getcwd()
        os.chdir(workdir)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                model = ReactionModel(setup_file='wgs.mkm')
                model.use_numbers_solver = False
                model.run()
                solver = model.solver
                solver.use_numbers_solver = False
                params = model.scaler.get_rxn_parameters([d['T'], 0.])
                try:  # initializes the solver for these parameters
                    solver.get_coverage(params, c0=[mpmath.mpf('1e-6')]
                                        * len(model.adsorbate_names))
                except Exception:
                    pass
                solver._rxn_parameters = params
        finally:
            os.chdir(cwd)
    if abs(float(model._kB) - ASE_KB) > 1e-20:
        raise RuntimeError('CatMap constants were not replaced')

    def rhs(t, y):
        cov = [mpmath.mpf(float(v)) for v in y]
        return [float(v) for v in solver.interacting_steady_state_function(cov)]

    n = len(model.adsorbate_names)
    sol = solve_ivp(rhs, (0., 1e3), np.zeros(n), method='Radau',
                    rtol=1e-11, atol=1e-22)
    cov = [mpmath.mpf(float(v)) for v in sol.y[:, -1]]
    rates = dict(zip(d['steps'], [float(r) for r in solver.get_rate(
        params, coverages=cov, verify_coverages=False)]))
    co2 = sum(rates[name] for name in CO2_STEPS)
    residual = max(abs(float(v)) for v in
                   solver.interacting_steady_state_function(cov))
    coverage = dict(zip(model.adsorbate_names, [float(v) for v in cov]))
    return coverage, co2, residual / co2


if __name__ == '__main__':
    results = []
    for d in json.load(open(INPUT)):
        coverage, co2, residual = solve(d)
        print('%-12s condition %2d: CO2 rate %.10e /s, steady-state '
              'residual / rate %.1e' % (d['variant'], d['condition'], co2,
                                        residual))
        results.append({'variant': d['variant'], 'condition': d['condition'],
                        'coverage': coverage, 'co2_rate': co2,
                        'residual': residual})
    with open(REFERENCE, 'w') as f:
        json.dump(results, f, indent=1)
        f.write('\n')
    print('Wrote', REFERENCE)
