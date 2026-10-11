"""Solve the electrochemical models of catmap_echem.py with CatMap 0.3.1.

Run in an environment with CatMap 0.3.1 (it needs numpy < 1.24, so not in
micki's environment), from the repository root, one model and potential
per task (each takes seconds to minutes; run them as batch jobs):

    python tests/catmap_echem_reference.py <model> <index>   # e.g. orr 3

writes catmap_echem_<model>_v<index>.json in the current directory, and

    python tests/catmap_echem_reference.py --merge <directory>

collects those into data/catmap_echem_reference.json.

The inputs are those of the CatMap templates (ThermodynamicScaler with the
voltage and temperature as descriptors, frozen free energies,
simple_electrochemical: G(pe_g) = -U, barriers '^x eV' at dG = 0 with the
transition state at G_FS + x + (1 - beta)(-dG)). CatMap's CODATA-2010 kB and
h are replaced by ASE's. CatMap's classic Newton solver is used if it
converges; otherwise CatMap's own d(theta)/dt is integrated with SciPy's
Radau solver from an empty surface and polished with CatMap's Newton solver
when that converges.
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
import catmap_echem  # noqa: E402

ASE_H = _hplanck / _e


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


def write_inputs(m, V, workdir):
    rows = ['surface_name\tsite_name\tspecies_name\tformation_energy\t'
            'bulk_structure\tfrequencies\tother_parameters\treference']
    for name, (site, E) in m['energies'].items():
        base = name.rsplit('_', 1)[0]
        if site == 'gas':
            rows.append('None\tgas\t%s\t%r\tNone\t[]\t[]\tx' % (base, E))
        else:
            rows.append('Pt\t%s\t%s\t%r\tfcc\t[]\t[]\tx' % (site, base, E))
    for site in m['sites']:
        rows.append('Pt\t%s\t*\t0.0\tfcc\t[]\t[]\tx' % site)
    with open(os.path.join(workdir, 'energies.txt'), 'w') as f:
        f.write('\n'.join(rows) + '\n')

    expressions = []
    for expression, prefactor, beta in m['reactions']:
        options = ['beta = %r' % beta]
        if prefactor is not None:
            options.insert(0, 'prefactor=%r' % prefactor)
        expressions.append(expression + '; ' + ', '.join(options))
    T = catmap_echem.T
    lines = [
        "scaler = 'ThermodynamicScaler'",
        'rxn_expressions = %r' % expressions,
        "surface_names = ['Pt']",
        "descriptor_names = ['voltage', 'temperature']",
        'descriptor_ranges = [[%r, %r], [%r, %r]]' % (V, V, T, T),
        'resolution = [1, 1]',
        'species_definitions = {}',
        "data_file = 'model.pkl'",
        "input_file = 'energies.txt'",
        "gas_thermo_mode = 'frozen_gas'",
        "adsorbate_thermo_mode = 'frozen_adsorbate'",
        "electrochemical_thermo_mode = 'simple_electrochemical'",
        'decimal_precision = 100',
        'tolerance = 1e-30',
        'max_rootfinding_iterations = 300',
        'max_bisections = 8',
    ]
    for site in m['sites']:
        lines.append("species_definitions[%r] = {'site_names': [%r], "
                     "'total': 1.0}" % (site, site))
    for name, p in m['pressures'].items():
        lines.append("species_definitions[%r] = {'pressure': %r}"
                     % (name, p))
    with open(os.path.join(workdir, 'model.mkm'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def run(m, V):
    T = catmap_echem.T
    with tempfile.TemporaryDirectory() as workdir:
        write_inputs(m, V, workdir)
        cwd = os.getcwd()
        os.chdir(workdir)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                model = ReactionModel(setup_file='model.mkm')
                model.use_numbers_solver = False
                model.output_variables += ['rate']
                try:
                    model.run()
                except Exception as e:
                    print('run failed:', e)
                solver = model.solver
                solver.use_numbers_solver = False
                params = model.scaler.get_rxn_parameters([V, T])
                native = None
                if model.coverage_map:
                    native = [mpmath.mpf(c) for c in model.coverage_map[0][1]]
        finally:
            os.chdir(cwd)
    if abs(float(model._kB) - ASE_KB) > 1e-20:
        raise RuntimeError('CatMap constants were not replaced')
    names = list(model.adsorbate_names)
    f_ss = solver.steady_state_function

    if native is not None:
        cov, method = native, 'catmap'
    else:
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
    G = model.scaler.get_free_energies([V, T])
    return {'V': V, 'method': method,
            'coverage': dict(zip(names, [float(c) for c in cov])),
            'rates': rates, 'kf': [float(x) for x in kf],
            'kr': [float(x) for x in kr], 'residual': residual,
            'G': {key: float(v) for key, v in G.items()}}


if __name__ == '__main__':
    if sys.argv[1] == '--merge':
        directory = sys.argv[2]
        ref = {}
        for key, m in catmap_echem.MODELS.items():
            ref[key] = [json.load(open(os.path.join(
                directory, 'catmap_echem_%s_v%d.json' % (key, i))))
                for i in range(len(m['voltages']))]
        with open(catmap_echem.REFERENCE, 'w') as f:
            json.dump(ref, f, indent=1)
            f.write('\n')
        print('Wrote', catmap_echem.REFERENCE)
    else:
        key, i = sys.argv[1], int(sys.argv[2])
        V = catmap_echem.MODELS[key]['voltages'][i]
        r = run(catmap_echem.MODELS[key], V)
        print('%s V=%g %s: rates %s, residual %.1e'
              % (key, V, r['method'],
                 ' '.join('%.6e' % x for x in r['rates']), r['residual']))
        with open('catmap_echem_%s_v%d.json' % (key, i), 'w') as f:
            json.dump(r, f, indent=1)
