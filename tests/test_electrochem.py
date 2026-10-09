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

from micki import (Adsorbate, Electron, Gas, Liquid, Lattice,  # noqa: E402
                   Model, Reaction, Solute, reactions_from_strings)
from micki.db import read_from_db  # noqa: E402

T = 298.15
NERNST = kB * T * np.log(10.)  # V per pH unit (0.05916 V at 298.15 K)


def redox_model(U=0., pH=0., reaction='a + h3o + e -> ah + h2o', **kw):
    """A* + H3O+ + e- <-> AH* + H2O, with dG = -0.4 eV + e U (U vs SHE)
    at 1 M H3O+ and pure water."""
    slab = Adsorbate(Atoms(), 'slab', E=0.)
    species = [slab, Electron('e'),
               Adsorbate(Atoms('O'), 'a', E=-0.5, sites=[slab]),
               Adsorbate(Atoms('OH'), 'ah', E=-0.9, sites=[slab]),
               Solute('h3o', 0., 'H3O', charge=1),
               Solute('h2o', 0., 'H2O', rhoref=55.5)]
    model = Model(T, Asite=1e-19, U_SHE=U, pH=pH)
    model.add_reactions(reactions_from_strings(species, {'r': (reaction, kw)}))
    model.set_fixed(['h3o', 'h2o'])
    model.set_initial_conditions({'h3o': 10.**-pH, 'h2o': 55.5, 'a': 0.5})
    return model


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


class PotentialTest(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore')

    def test_electron(self):
        e = Electron('e', dE=0.05)
        e.U = 0.3
        self.assertAlmostEqual(e.get_G(T), -0.25, places=14)
        self.assertEqual(e.charge, -1)
        self.assertEqual(e.copy('e2').get_G(T), e.get_G(T))

    def test_electron_is_not_a_variable(self):
        model = redox_model()
        self.assertNotIn('e', model._variable_labels)
        rate = model.rates[0]
        self.assertNotIn('e', {str(x) for x in rate.free_symbols})
        self.assertEqual(model.reactions['r'].n_electrons, 1)

    def test_nernst(self):
        # theta_AH / theta_A = exp(-(dG(U) - kT ln[H3O+]) / kT): the
        # half-reduction potential shifts by -59.16 mV per pH unit (vs SHE)
        # and is independent of pH vs RHE
        for pH in (0., 1., 3.):
            for U in (0.25, 0.4, 0.55):
                model = redox_model(U=U - NERNST * pH, pH=pH)
                _, cov, _ = model.find_steady_state()
                ratio = cov['ah'] / cov['a']
                expected = np.exp((0.4 - U) / (kB * T))
                self.assertAlmostEqual(ratio / expected, 1., places=8,
                                       msg='pH {}, U_RHE {}'.format(pH, U))
                self.assertAlmostEqual(model.U_RHE, U, places=12)

    def test_potential_properties(self):
        model = redox_model(U=0.1, pH=2.)
        with self.assertRaises(AttributeError):
            model.U_RHE = 0.3
        model.set_potential(0.5, scale='RHE')
        self.assertAlmostEqual(model.U_SHE, 0.5 - 2 * NERNST, places=14)
        model.set_potential(0.2)
        self.assertEqual(model.U_SHE, 0.2)
        with self.assertRaises(ValueError):
            model.set_potential(0.2, scale='RHE x')
        with self.assertRaisesRegex(ValueError, 'pH'):
            Model(T, 1e-19).U_RHE
        # changing U rebuilds: same steady state as a model built at that U
        model.U_SHE = 0.35
        _, cov, r = model.find_steady_state()
        _, cov_new, r_new = redox_model(U=0.35, pH=2.).find_steady_state()
        self.assertAlmostEqual(cov['ah'] / cov_new['ah'], 1., places=10)
        self.assertEqual(model.copy().U_SHE, 0.35)


class BarrierTest(unittest.TestCase):
    """Potential-dependent barriers of electrochemical steps."""

    def setUp(self):
        warnings.simplefilter('ignore')
        self.slab = Adsorbate(Atoms(), 'slab', E=0.)
        self.sp = {sp.label: sp for sp in [
            self.slab, Electron('e'),
            Adsorbate(Atoms('O'), 'a', E=-0.5, sites=[self.slab]),
            Adsorbate(Atoms('OH'), 'ah', E=-0.9, sites=[self.slab]),
            # transition state at U_ref = 0.2 V: barrier 0.5 eV there
            Adsorbate(Atoms('O2H3'), 'ts', E=-0.2, ts=True,
                      sites=[self.slab]),
            Solute('h3o', 0., 'H3O', charge=1),
            Solute('h2o', 0., 'H2O', rhoref=55.5)]}

    def _at(self, rxn, U):
        rxn.update(T=T, Asite=1e-19, L=0, U=U)
        return float(rxn.dG_act), float(rxn.dG), float(rxn.kfor), \
            float(rxn.krev)

    def test_transition_state_at_U_ref(self):
        # dG_act(U) = dG_act(U_ref) + beta (dG(U) - dG(U_ref)); the reverse
        # barrier changes by -(1 - beta) per volt; alpha defaults to beta
        for kw in ({'ts': self.sp['ts']}, {'dG_act': 0.5}):
            rxn = Reaction.from_string('a + h3o + e -> ah + h2o', self.sp,
                                       U_ref=0.2, beta=0.3, **kw)
            self.assertEqual(rxn.clip, 'coverage')
            ref_act, ref_dG, _, _ = self._at(rxn, 0.2)
            self.assertAlmostEqual(ref_act, 0.5, places=12)
            for U in (-0.1, 0.5):
                act, dG, kf, kr = self._at(rxn, U)
                self.assertAlmostEqual(dG - ref_dG, U - 0.2, places=12)
                self.assertAlmostEqual(act - ref_act, 0.3 * (U - 0.2),
                                       places=12)
                self.assertAlmostEqual((act - dG) - (ref_act - ref_dG),
                                       -0.7 * (U - 0.2), places=12)
                self.assertAlmostEqual(
                    kf / (kB * T / 4.135667696e-15 * np.exp(-act / (kB * T))),
                    1., places=6)
        self.assertEqual(Reaction.from_string(
            'a + h3o + e <-> ts -> ah + h2o', self.sp, U_ref=0.2,
            beta=0.3).alpha_fixed, 0.3)

    def test_simple_pcet(self):
        # dG_act = dG_act0 + beta dG (CatMap's G_TS = G_FS + barrier
        # + (1 - beta)(-dG)), clipped at max(dG, 0)
        rxn = Reaction.from_string('a + h3o + e <-> ^0.26 -> ah + h2o',
                                   self.sp)
        self.assertEqual((rxn.dG_act0, rxn.beta, rxn.method),
                         (0.26, 0.5, 'TST'))
        kT_h = kB * T / 4.135667696e-15
        for U in (0.1, 0.4, 0.8):
            act, dG, kf, kr = self._at(rxn, U)
            G_ts = (-0.9 + 0.) + 0.26 + 0.5 * (-dG)  # CatMap, FS = ah + h2o
            G_is = -0.5 - U
            self.assertAlmostEqual(act, G_ts - G_is, places=12)
            self.assertAlmostEqual(kf / (kT_h * np.exp(-max(act, dG, 0.)
                                                       / (kB * T))), 1.,
                                   places=6)
        # strongly reducing: the barrier would be negative, so it is 0
        act, dG, kf, kr = self._at(rxn, -0.3)
        self.assertLess(act, 0.)
        self.assertAlmostEqual(kf / kT_h, 1., places=6)

    def test_tafel_slope_of_rate_constant(self):
        # d ln kfor / dU = -beta n e / kT
        rxn = Reaction.from_string('a + h3o + e <-> ts -> ah + h2o', self.sp,
                                   U_ref=0.2, beta=0.35)
        kf1 = self._at(rxn, 0.20)[2]
        kf2 = self._at(rxn, 0.25)[2]
        self.assertAlmostEqual(np.log(kf2 / kf1) / 0.05, -0.35 / (kB * T),
                               places=8)

    def test_reorganization_and_prefactor(self):
        base = Reaction.from_string('a + h3o + e <-> ^0.26 -> ah + h2o',
                                    self.sp)
        reorg = Reaction.from_string('a + h3o + e <-> ^0.26 -> ah + h2o',
                                     self.sp, dG_reorg=0.22)
        pref = Reaction.from_string('a + h3o + e <-> ^0.26 -> ah + h2o',
                                    self.sp, prefactor=1e9)
        _, _, kf, kr = self._at(base, 0.5)
        _, _, kf_r, kr_r = self._at(reorg, 0.5)
        act, _, kf_p, _ = self._at(pref, 0.5)
        self.assertAlmostEqual(kf_r / kf, np.exp(-0.22 / (kB * T)), places=12)
        self.assertAlmostEqual(kr_r / kr, np.exp(-0.22 / (kB * T)), places=12)
        self.assertAlmostEqual(kf_p / (1e9 * np.exp(-act / (kB * T))), 1.,
                               places=12)
        # a barrierless (EQUIL) adsorption step gets it too
        eq = Reaction(self.sp['h2o'] + self.slab,
                      Adsorbate(Atoms('H2O'), 'h2o*', E=0.1,
                                sites=[self.slab]), dG_reorg=0.28)
        eq0 = Reaction(self.sp['h2o'] + self.slab, eq.products[0])
        for r in (eq, eq0):
            r.update(T=T, Asite=1e-19, L=0)
        self.assertAlmostEqual(float(eq.kfor / eq0.kfor),
                               np.exp(-0.28 / (kB * T)), places=12)

    def test_barrier_token(self):
        for token in ('^0.26', '^ 0.26 eV', '^0.26eV_a', '^2.6e-1'):
            rxn = Reaction.from_string(
                'a + h3o + e <-> {} -> ah + h2o'.format(token), self.sp)
            self.assertAlmostEqual(rxn.dG_act0, 0.26, places=14)
        with self.assertRaisesRegex(ValueError, 'barrier'):
            Reaction.from_string('a + h3o + e <-> ^x -> ah + h2o', self.sp)
        with self.assertRaisesRegex(ValueError, 'twice'):
            Reaction.from_string('a + h3o + e <-> ^0.2 -> ah + h2o', self.sp,
                                 dG_act0=0.3)

    def test_errors_and_defaults(self):
        sp = self.sp
        with self.assertRaisesRegex(ValueError, 'U_ref'):
            Reaction.from_string('a + h3o + e <-> ts -> ah + h2o', sp)
        with self.assertRaisesRegex(ValueError, 'U_ref'):
            Reaction(sp['a'], sp['a'], dG_act=0.3, U_ref=0.1)
        with self.assertRaisesRegex(ValueError, 'beta'):
            Reaction(sp['a'], sp['a'], dG_act0=0.3)
        with self.assertRaisesRegex(ValueError, 'beta'):
            Reaction.from_string('a + h3o + e -> ah + h2o', sp, beta=1.5)
        with self.assertRaisesRegex(ValueError, 'prefactor'):
            Reaction.from_string('a + h3o + e -> ah + h2o', sp,
                                 method='DIEQUIL', prefactor=1e9)
        # thermal steps keep micki's defaults; explicit arguments win
        thermal = Reaction(sp['a'], sp['a'], dG_act0=0.3, beta=0.5)
        self.assertIsNone(thermal.clip)
        rxn = Reaction.from_string('a + h3o + e -> ah + h2o', sp, clip=None)
        self.assertIsNone(rxn.clip)


if __name__ == '__main__':
    unittest.main()
