"""micki with CatMap's conventions against CatMap's own solution.

The WGS model in CatMap's conventions (catmap_wgs.py: 1 bar gas reference,
clip='coverage', barrierless and non-activated adsorption, first-order
interactions with linear or piecewise-linear response, fixed or explicit
transition-state interactions) was solved with CatMap 0.3.1
(catmap_reference.py); micki reproduces its CO2 rates and coverages to
~1e-13.

catmap_models.py has two more models (ammonia synthesis; ethylene
hydrogenation with two site types) that CatMap solved natively, with its
own thermochemistry and interaction model (catmap_models_reference.py);
micki, computing all thermochemistry itself, reproduces CatMap's free
energies, rate constants, coverages and rates.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

import json
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import catmap_models  # noqa: E402
import catmap_wgs  # noqa: E402
import numpy as np  # noqa: E402
import sympy as sym  # noqa: E402
from ase.units import _amu, _e  # noqa: E402
from micki import Gas, Model  # noqa: E402
from micki.utils import bar_to_molar  # noqa: E402


class CatMapTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter('ignore')
        with open(catmap_wgs.REFERENCE) as f:
            cls.reference = json.load(f)
        with open(catmap_wgs.INPUT) as f:
            cls.inputs = json.load(f)

    def _assert_close(self, a, b, rtol, msg):
        self.assertLessEqual(abs(a - b), rtol * abs(b), msg)

    def test_inputs_match_model(self):
        # the CatMap reference was computed for the current model
        for d in self.inputs:
            new = catmap_wgs.export(d['variant'], d['condition'])
            msg = '{} {}'.format(d['variant'], d['condition'])
            for name, E in d['energies'].items():
                self._assert_close(new['energies'][name], E, 1e-12, msg)
            for name, step in d['steps'].items():
                if step['ts'] is not None:
                    self.assertAlmostEqual(new['steps'][name]['E_ts'],
                                           step['E_ts'], places=12, msg=msg)
                    self.assertEqual(new['steps'][name]['eps'], step['eps'])
            self.assertEqual(new['eps'], d['eps'])
            self.assertEqual(new['pressures'], d['pressures'])

    def test_steady_states(self):
        for d in self.reference:
            msg = '{} {}'.format(d['variant'], d['condition'])
            coverage, co2 = catmap_wgs.solve(d['variant'], d['condition'])
            self._assert_close(co2, d['co2_rate'], 1e-10, msg)
            for name, catmap_name in catmap_wgs.ADS.items():
                theta = d['coverage'][catmap_name]
                if theta > 1e-12:
                    self._assert_close(coverage[name], theta, 1e-10,
                                       msg + ' ' + name)



class CatMapModelsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter('ignore')
        with open(catmap_models.REFERENCE) as f:
            cls.reference = json.load(f)

    @staticmethod
    def _at(expr, values):
        expr = sym.sympify(expr)
        return float(expr.subs({s: values.get(s.name, 0.)
                                for s in expr.free_symbols}))

    def test_models(self):
        # CatMap's collision-theory prefactor, 1/sqrt(m u e kT), uses
        # CODATA-2010 values of the atomic mass unit and the electron volt
        codata2010 = np.sqrt(_amu * _e / (1.660538921e-27 * 1.602176565e-19))
        for key, m in catmap_models.MODELS.items():
            for c in self.reference[key]:
                T, msg = c['T'], '{} T={}'.format(key, c['T'])
                sp, sites, rxns = catmap_models.build(
                    m, c['interaction_matrix'])
                # free energies at zero coverage, gases at 1 bar
                for name, species in sp.items():
                    self.assertAlmostEqual(self._at(species.get_G(T), {}),
                                           c['G'][name], places=12,
                                           msg=msg + ' ' + name)
                model = Model(T, m['A_site'] * 1e-20, reactor='CSTR')
                model.add_reactions(rxns)
                model.set_fixed(list(m['gases']))
                # rate constants at CatMap's steady state, CatMap's units
                c0 = bar_to_molar(1., T)
                for i, rxn in enumerate(rxns.values()):
                    n_r = sum(isinstance(s, Gas) for s in rxn.reactants)
                    n_p = sum(isinstance(s, Gas) for s in rxn.products)
                    f = codata2010 if rxn.method == 'STICK' else 1.
                    kf = self._at(rxn.kfor, c['coverage']) * c0 ** n_r * f
                    kr = self._at(rxn.krev, c['coverage']) * c0 ** n_p * f
                    for k, ref in ((kf, c['kf'][i]), (kr, c['kr'][i])):
                        self.assertLessEqual(abs(k / ref - 1), 1e-11,
                                             '{} r{}'.format(msg, i))
                # steady state
                U0 = {g: bar_to_molar(v, T) for g, v in c['p'].items()}
                U0.update({a: 0. for a in m['adsorbates']})
                model.set_initial_conditions(U0)
                _, U, r = model.find_steady_state()
                for a, theta in c['coverage'].items():
                    if theta > 1e-10:
                        self.assertLessEqual(abs(float(U[a]) / theta - 1),
                                             1e-9, msg + ' ' + a)
                # net rates of nearly equilibrated steps are differences of
                # much larger fluxes, only meaningful to ~1e-16 * flux/net
                # in double precision
                fluxes = catmap_models.forward_fluxes(m, c)
                for i, (ref, flux) in enumerate(zip(c['rates'], fluxes)):
                    if flux < 1e7 * abs(ref):
                        rate = float(r['r%d' % i])
                        self.assertLessEqual(abs(rate / ref - 1), 1e-6,
                                             '{} r{}'.format(msg, i))


if __name__ == '__main__':
    unittest.main()
