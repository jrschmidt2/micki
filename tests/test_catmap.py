"""micki with CatMap's conventions against CatMap's own solution.

The WGS model in CatMap's conventions (catmap_wgs.py: 1 bar gas reference,
clip='coverage', barrierless and non-activated adsorption, first-order
interactions with linear or piecewise-linear response, fixed or explicit
transition-state interactions) was solved with CatMap 0.3.1
(catmap_reference.py); micki reproduces its CO2 rates and coverages to
~1e-13.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

import json
import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import catmap_wgs  # noqa: E402


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


if __name__ == '__main__':
    unittest.main()
