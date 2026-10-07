"""Unit tests for Model, Reaction and the species classes.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wgs  # noqa: E402
from micki import Model, Reaction  # noqa: E402


class ModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter('ignore')
        cls.sp = wgs.build_species()
        cls.rxns = wgs.build_reactions(cls.sp)

    def test_add_reactions_skips_known_reactions(self):
        model = Model(548, wgs.ASITE)
        model.add_reactions({'co_ads': self.rxns['co_ads']})
        model.add_reactions({'co_ads': self.rxns['co_ads'],
                             'h2o_ads': self.rxns['h2o_ads']})
        self.assertEqual(list(model.reactions), ['co_ads', 'h2o_ads'])

    def test_set_scale_rejects_unknown_parameters(self):
        rxn = Reaction(self.sp['co_g'], self.sp['co'], method='STICK')
        for name in ['kfwd', 'dH', 'dS']:
            with self.assertRaises(ValueError):
                rxn.set_scale(name, 2.)
            with self.assertRaises(ValueError):
                rxn.get_scale(name)
        rxn.set_scale('kfor', 2.)
        self.assertEqual(rxn.get_scale('kfor'), 2.)


if __name__ == '__main__':
    unittest.main()
