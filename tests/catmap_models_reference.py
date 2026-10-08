"""Solve the models of catmap_models.py natively with CatMap 0.3.1.

Run in an environment with CatMap 0.3.1 (it needs numpy < 1.24, so not in
micki's environment), from the repository root. Each condition takes
1-6 minutes (CatMap evaluates its rate equations in 100-digit mpmath), so
run it as a batch job, one condition per task:

    python tests/catmap_models_reference.py <model> <index>   # e.g. nh3 0

writes catmap_<model>_c<index>.json in the current directory, and

    python tests/catmap_models_reference.py --merge <directory>

collects those into data/catmap_models_reference.json.

Thermochemistry by CatMap (ideal_gas with its geometries and symmetry
numbers, harmonic_adsorbate), interactions by CatMap (first_order,
geometric-mean cross terms, intermediate_state TS weighting). CatMap's
CODATA-2010 kB and h are replaced by ASE's; the CODATA-2010 atomic mass
unit and electron volt in its non-activated (collision theory) prefactor
remain (-5.3e-8 in those rate constants). CatMap's classic Newton solver
is used if it converges; otherwise CatMap's own d(theta)/dt is integrated
with SciPy's Radau solver from an empty surface and polished with
CatMap's Newton solver when that converges.
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import warnings

import mpmath
import numpy as np
from ase.units import kB as ASE_KB, _hplanck, _e
from scipy.integrate import solve_ivp

warnings.simplefilter('ignore')
from catmap import ReactionModel  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catmap_models  # noqa: E402

ASE_H = _hplanck / _e
SITE_NAMES = {'s': '111', 'h': 'hollow'}


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


def write_inputs(m, T, p, workdir):
    rows = ['surface_name\tsite_name\tspecies_name\tformation_energy\t'
            'bulk_structure\tfrequencies\tother_parameters\treference']
    for g, E in m['gases'].items():
        rows.append('None\tgas\t%s\t%r\tNone\t%r\t[]\tx'
                    % (g[:-2], E, catmap_models.GAS_FREQS[g]))
    for name, (E, freqs) in list(m['adsorbates'].items()) + \
            list(m['ts'].items()):
        base, site = name.rsplit('_', 1)
        rows.append('Pt\t%s\t%s\t%r\tfcc\t%r\t[]\tx'
                    % (SITE_NAMES[site], base, E, freqs))
    with open(os.path.join(workdir, 'energies.txt'), 'w') as f:
        f.write('\n'.join(rows) + '\n')

    prefactors = [None if pf is None else
                  {'type': pf, 'A_site': m['A_site']}
                  for _, pf in m['reactions']]
    lines = [
        "scaler = 'ThermodynamicScaler'",
        'rxn_expressions = %r' % [e for e, _ in m['reactions']],
        'prefactor_list = %r' % prefactors,
        "surface_names = ['Pt']",
        "descriptor_names = ['temperature', 'logPressure']",
        'descriptor_ranges = [[%r, %r], [0, 0]]' % (T, T),
        'resolution = 1',
        'species_definitions = {}',
        "data_file = 'model.pkl'",
        "input_file = 'energies.txt'",
        "gas_thermo_mode = 'ideal_gas'",
        "adsorbate_thermo_mode = 'harmonic_adsorbate'",
        'decimal_precision = 100',
        'tolerance = 1e-30',
        'max_rootfinding_iterations = 300',
        'max_bisections = 8',
    ]
    for site in m['sites']:
        lines.append("species_definitions[%r] = {'site_names': [%r], "
                     "'total': 1}" % (site, SITE_NAMES[site]))
    for g, pressure in p.items():
        lines.append("species_definitions[%r] = {'concentration': %r}"
                     % (g, pressure))
    inter = m['interactions']
    if inter:
        lines += [
            "adsorbate_interaction_model = 'first_order'",
            'interaction_response_function = %r' % inter['response'],
            'interaction_response_parameters = %r' % inter['params'],
            'cross_interaction_mode = %r' % inter['cross_mode'],
            "transition_state_cross_interaction_mode = 'intermediate_state'",
        ]
        for site in m['sites']:
            lines.append("species_definitions[%r]['interaction_response_"
                         "parameters'] = %r" % (site, inter['params']))
        for name, eps in inter['self'].items():
            lines.append("species_definitions[%r] = {'self_interaction_"
                         "parameter': [%r]}" % (name, eps))
    with open(os.path.join(workdir, 'model.mkm'), 'w') as f:
        f.write('\n'.join(lines) + '\n')




def run(m, T, p):
    with tempfile.TemporaryDirectory() as workdir:
        write_inputs(m, T, p, workdir)
        cwd = os.getcwd()
        os.chdir(workdir)
        log = io.StringIO()
        try:
            with contextlib.redirect_stdout(log):
                model = ReactionModel(setup_file='model.mkm')
                # CatMap's classic Newton solver (the numbers solver works in
                # other variables, which the integration below cannot use)
                model.use_numbers_solver = False
                model.output_variables += ['rate']
                try:
                    model.run()
                except Exception as e:
                    print('run failed:', e)
                solver = model.solver
                solver.use_numbers_solver = False
                params = model.scaler.get_rxn_parameters([T, 0.])
                native = None
                if model.coverage_map:
                    native = [mpmath.mpf(c) for c in model.coverage_map[0][1]]
        finally:
            os.chdir(cwd)
    if abs(float(model._kB) - ASE_KB) > 1e-20:
        raise RuntimeError('CatMap constants were not replaced')
    names = list(model.adsorbate_names)
    interacting = bool(m['interactions'])
    f_ss = (solver.interacting_steady_state_function if interacting
            else solver.steady_state_function)

    if native is not None:
        cov, method = native, 'catmap'
    else:
        # integrate CatMap's own d(theta)/dt from an empty surface
        solver._rxn_parameters = params
        try:
            solver.get_coverage(params, c0=[mpmath.mpf('1e-6')] * len(names))
        except Exception:
            pass
        solver._rxn_parameters = params

        def rhs(t, y):
            c = [mpmath.mpf(float(v)) for v in y]
            solver.get_rate_constants(params, c)
            return [float(v) for v in f_ss(c)]
        sol = solve_ivp(rhs, (0., 1e6), np.zeros(len(names)),
                        method='Radau', rtol=1e-11, atol=1e-22)
        cov = [mpmath.mpf(float(v)) for v in sol.y[:, -1]]
        method = 'integrated'
        # polish with CatMap's own root finder
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                polished = solver.get_coverage(params, c0=cov)
            if polished is not None:
                cov, method = list(polished), 'integrated+catmap newton'
        except Exception:
            pass
    solver._rxn_parameters = params
    solver.get_rate_constants(params, cov)
    rates = [float(r) for r in solver.get_rate(params, coverages=cov,
                                                verify_coverages=False)]
    k = solver.get_rate_constants(params, cov)
    kf, kr = k[:len(k) // 2], k[len(k) // 2:]
    residual = max(abs(float(v)) for v in f_ss(cov))
    G = model.scaler.get_free_energies([T, 0.])
    out = {'T': T, 'p': p, 'method': method,
           'coverage': dict(zip(names, [float(c) for c in cov])),
           'rates': rates, 'kf': [float(k) for k in kf],
           'kr': [float(k) for k in kr], 'residual': residual,
           'G': {k: float(v) for k, v in G.items()},
           'adsorbate_names': names,
           'ts_names': list(model.transition_state_names)}
    if interacting:
        ia = model.thermodynamics.adsorbate_interactions
        # the matrix CatMap last built for its rate expressions
        mat = ia._interaction_matrix
        all_names = names + list(model.transition_state_names)
        out['interaction_matrix'] = {
            a: {b: float(mat[i][j]) for j, b in enumerate(all_names)}
            for i, a in enumerate(all_names)}
    return out


if __name__ == '__main__':
    if sys.argv[1] == '--merge':
        directory = sys.argv[2]
        ref = {}
        for key, m in catmap_models.MODELS.items():
            ref[key] = [json.load(open(os.path.join(
                directory, 'catmap_%s_c%d.json' % (key, i))))
                for i in range(len(m['conditions']))]
        with open(catmap_models.REFERENCE, 'w') as f:
            json.dump(ref, f, indent=1)
            f.write('\n')
        print('Wrote', catmap_models.REFERENCE)
    else:
        key, i = sys.argv[1], int(sys.argv[2])
        T, p = catmap_models.MODELS[key]['conditions'][i]
        r = run(catmap_models.MODELS[key], T, p)
        print('%s T=%.0f %s: rates %s, residual %.1e'
              % (key, T, r['method'],
                 ' '.join('%.6e' % x for x in r['rates']), r['residual']))
        with open('catmap_%s_c%d.json' % (key, i), 'w') as f:
            json.dump(r, f, indent=1)
