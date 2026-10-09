"""Species with given energies and charges, and electrochemistry.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest
import warnings

import numpy as np
from ase import Atoms
from ase.db import connect
from ase.units import kB

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micki import Adsorbate, Gas, Liquid, Lattice, Reaction, Solute  # noqa: E402
from micki.db import read_from_db  # noqa: E402


class GivenEnergyTest(unittest.TestCase):
    """E= and S= instead of computed thermochemistry."""

    def setUp(self):
        warnings.simplefilter('ignore')
        self.slab = Adsorbate(Atoms(), 'slab', E=0.)

    def test_free_energies(self):
        # no structure, calculator or frequencies needed; G = E - T S, plus
        # kT for gases only; dE adds
        ads = Adsorbate(Atoms('CO'), 'co', E=-1.0, S=1e-3, sites=[self.slab],
                        dE=0.1)
        gas = Gas(Atoms('CO'), 'co_g', E=0.5, S=2e-3)
        liquid = Liquid(Atoms('CO'), 'co_l', E=0.2, S=1e-3)
        solute = Solute('oh', -1.0, 'OH', S=5e-4, charge=-1)
        for T in (300., 600.):
            self.assertAlmostEqual(ads.get_G(T), -0.9 - T * 1e-3, places=12)
            self.assertAlmostEqual(gas.get_G(T), 0.5 + kB * T - T * 2e-3,
                                   places=12)
            self.assertAlmostEqual(liquid.get_G(T), 0.2 - T * 1e-3,
                                   places=12)
            self.assertAlmostEqual(solute.get_G(T), -1.0 - T * 5e-4,
                                   places=12)
            self.assertAlmostEqual(ads.get_q(T),
                                   np.exp(-ads.get_G(T) / (kB * T)),
                                   places=12)
        # E alone is a free energy
        self.assertEqual(Adsorbate(Atoms('O'), 'o', E=-0.3,
                                   sites=[self.slab]).get_G(500.), -0.3)
        with self.assertRaisesRegex(ValueError, 'S requires E'):
            Gas(Atoms('CO'), 'x', freqs=[0.] * 5 + [0.26], S=1e-3)

    def test_equilibrium_constant(self):
        T = 400.
        gas = Gas(Atoms('CO'), 'co_g', E=0.)
        ads = Adsorbate(Atoms('CO'), 'co', E=-1.2, S=-1e-3, sites=[self.slab])
        rxn = Reaction(gas, ads)
        rxn.update(T=T, Asite=1e-19, L=0)
        dG = ads.get_G(T) - gas.get_G(T)
        self.assertAlmostEqual(float(rxn.keq) / np.exp(-dG / (kB * T)), 1.,
                               places=12)

    def test_lattice_and_copy(self):
        # configurational entropy still adds to a given E; copies keep E, S
        ts = Adsorbate(Atoms('O2'), 'o-o', E=0.4, S=1e-4, ts=True,
                       sites=[self.slab, self.slab],
                       lattice=Lattice({self.slab: {self.slab: 6}}))
        self.assertAlmostEqual(ts.get_S(500.), 1e-4 + kB * np.log(6),
                               places=14)
        for species in (ts, Solute('h3o', 0., 'H3O', charge=1, rhoref=1.),
                        Gas(Atoms('H2'), 'h2', E=0.1, S=1e-3, charge=0)):
            copy = species.copy()
            self.assertEqual((copy.E_given, copy.S_given, copy.charge),
                             (species.E_given, species.S_given,
                              species.charge))
            self.assertAlmostEqual(copy.get_G(500.), species.get_G(500.),
                                   places=12)

    def test_database(self):
        species = [self.slab,
                   Adsorbate(Atoms('CO'), 'co', E=-1.0, S=1e-3,
                             sites=[self.slab]),
                   Gas(Atoms('CO'), 'co_g', E=0.5, S=2e-3),
                   Solute('h2o', -0.1, 'H2O', S=7e-4, rhoref=55.5),
                   Solute('oh', -1.0, 'OH', charge=-1)]
        with tempfile.TemporaryDirectory() as tmp:
            db = connect(os.path.join(tmp, 'species.json'))
            for sp in species:
                sp.save_to_db(db)
            read = read_from_db(db)
        for sp in species:
            new = read[sp.label]
            self.assertEqual(new.charge, sp.charge, sp.label)
            self.assertAlmostEqual(new.get_G(450.), sp.get_G(450.),
                                   places=12, msg=sp.label)
            self.assertEqual(new.get_reference_state(),
                             sp.get_reference_state(), sp.label)


class ChargeBalanceTest(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore')
        self.slab = Adsorbate(Atoms(), 'slab', E=0.)
        self.h3o = Solute('h3o', 0., 'H3O', charge=1)
        self.oh = Solute('oh', 0.8, 'OH', charge=-1)
        self.h2o = Solute('h2o', 0., 'H2O', rhoref=55.5)
        self.h = Adsorbate(Atoms('H'), 'h', E=-0.2, sites=[self.slab])

    def test_balanced(self):
        Reaction(self.h3o + self.oh, 2 * self.h2o)

    def test_unbalanced(self):
        # atoms balance, charge does not: H3O+ -> H2O + H*
        with self.assertRaisesRegex(ValueError,
                                    r'conserve charge \(reactants \+1, '
                                    r'products \+0\)'):
            Reaction(self.h3o, self.h2o + self.h)
        Reaction(self.h3o, self.h2o + self.h, check_balance=False)
        # a transition state with the wrong charge
        ts = Adsorbate(Atoms('H3O'), 'h3o-ts', E=0.5, ts=True, charge=0,
                       sites=[self.slab])
        with self.assertRaisesRegex(ValueError, 'transition state'):
            Reaction(self.h3o, self.h3o, ts=ts)


if __name__ == '__main__':
    unittest.main()
