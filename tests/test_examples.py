"""The examples in examples/ build the same models as the tests and run.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

import importlib.util
import json
import os
import sys
import unittest
import warnings

import sympy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wgs  # noqa: E402

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'examples')


def load(name):
    spec = importlib.util.spec_from_file_location(
        'example_' + name, os.path.join(EXAMPLES, name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WGSExampleTest(unittest.TestCase):
    """examples/wgs.py (reaction strings) is the model of tests/wgs.py."""

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter('ignore')
        cls.example = load('wgs')

    def test_same_model(self):
        sp_ex = self.example.build_species()
        sp = wgs.build_species()
        for name, species in sp.items():
            self.assertEqual(
                sympy.simplify(sympy.sympify(sp_ex[name].lateral)
                               - sympy.sympify(species.lateral)), 0, name)
            self.assertEqual(sp_ex[name].dE, species.dE, name)
            self.assertEqual(sp_ex[name].symm, species.symm, name)
        rxns_ex = self.example.build_reactions(sp_ex)
        rxns = wgs.build_reactions(sp)
        self.assertEqual(list(rxns_ex), list(rxns))
        for name in rxns:
            a, b = rxns_ex[name], rxns[name]
            self.assertEqual(a.method, b.method, name)
            for side in ('reactants', 'products'):
                self.assertEqual(sorted(map(str, getattr(a, side))),
                                 sorted(map(str, getattr(b, side))), name)
            self.assertEqual([str(s) for s in a.ts or []],
                             [str(s) for s in b.ts or []], name)
        self.assertEqual([c[:6] for c in self.example.CONDITIONS],
                         [tuple(c) for c in wgs.CONDITIONS])

    def test_turnover_frequencies(self):
        with open(wgs.REFERENCE) as f:
            reference = json.load(f)
        sp = self.example.build_species()
        reactions = self.example.build_reactions(sp)
        for i in (0, 12):
            tof = self.example.turnover_frequency(
                self.example.CONDITIONS[i], sp, reactions)
            self.assertAlmostEqual(tof / reference[i]['tof'], 1., places=6)


if __name__ == '__main__':
    unittest.main()
