"""Unit tests for Model, Reaction and the species classes.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest
import warnings

import numpy as np
import sympy
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wgs  # noqa: E402
import micki  # noqa: E402
from micki.masses import masses  # noqa: E402
from micki import (Model, Reaction, Gas, Liquid, Electron,  # noqa: E402
                   EnergyReference, Lattice)
from micki.db import read_from_db  # noqa: E402
from micki.utils import calculate_avg_vdw_radius, bar_to_molar  # noqa: E402
from ase.units import kB, _k, _hplanck  # noqa: E402


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

    def test_species_copy(self):
        T = 548
        gas = self.sp['co_g']
        liquid = Liquid(gas.atoms, 'co_l', gas.freqs, D=1e-9)
        electron = Electron(0.1, 0.05, 'e')
        for species in (gas, liquid, electron):
            new = species.copy(newlabel=species.label + '_copy')
            self.assertEqual(new.label, species.label + '_copy')
            self.assertEqual(type(new), type(species))
        self.assertAlmostEqual(gas.copy().get_G(T), gas.get_G(T), places=12)
        self.assertEqual(electron.copy().self_repulsion, 0.05)

    def test_get_G_without_temperature(self):
        gas = self.sp['co_g']
        gas.update(T=548)
        self.assertEqual(gas.get_G(), gas.get_G(548))

    def test_set_lattice_after_add_reactions(self):
        model = Model(548, wgs.ASITE)
        model.add_reactions(self.rxns)
        model.lattice = {self.sp['slab']: {self.sp['slab']: 6}}
        self.assertIsInstance(model.lattice, Lattice)
        self.assertIs(self.sp['co'].lattice, model.lattice)

    def test_reaction_does_not_modify_passed_reactants(self):
        products = self.sp['co_g'] + self.sp['co_g']
        Reaction(self.sp['co'] + self.sp['co'], products)
        self.assertEqual(len(products), 2)

    def test_stick_without_fluid(self):
        rxn = Reaction(self.sp['co'], self.sp['co'], method='STICK')
        with self.assertRaisesRegex(ValueError, 'STICK requires a fluid'):
            rxn.update(T=548, Asite=wgs.ASITE, L=0)

    def test_save_to_db_with_connection(self):
        from ase.db import connect
        with tempfile.TemporaryDirectory() as tmp:
            db = connect(os.path.join(tmp, 'species.json'))
            for name in ('slab', 'co'):
                species = self.sp[name]
                lateral, species.lateral = species.lateral, 0.
                try:
                    species.save_to_db(db)
                finally:
                    species.lateral = lateral
            read = read_from_db(db)
        np.testing.assert_allclose(read['co'].freqs, self.sp['co'].freqs)
        self.assertEqual([s.label for s in read['co'].sites], ['slab'])

    def test_adsorbates_do_not_share_default_sites(self):
        from micki import Adsorbate
        atoms, freqs = self.sp['co'].atoms, self.sp['co'].freqs
        self.assertIsNot(Adsorbate(atoms, 'x1', freqs).sites,
                         Adsorbate(atoms, 'x2', freqs).sites)

    def test_imaginary_frequencies_raise(self):
        from micki import Adsorbate
        freqs = -abs(self.sp['co'].freqs)
        with self.assertRaises(ValueError):
            Adsorbate(self.sp['co'].atoms, 'bad', freqs)

    def test_adsorption_equilibrium_constant(self):
        # K = Q_ads / Q_gas at the 1 M reference state (Hermes thesis eq.
        # 3.23), computed independently with ASE: the gas free energy must
        # be the ideal-gas chemical potential (including the pV = kT term)
        from ase.thermochemistry import IdealGasThermo, HarmonicThermo
        from ase.units import kB, _k, _Nav
        T = 548.
        sp = wgs.build_species()
        for s in sp.values():
            s.lateral = 0.
        for gas_name, ads_name in [('co_g', 'co'), ('h2o_g', 'h2o')]:
            gas, ads = sp[gas_name], sp[ads_name]
            rxn = Reaction(gas, ads, method='STICK')
            rxn.update(T=T, Asite=wgs.ASITE, L=0)
            atoms = gas.atoms.copy()
            atoms.pbc = False
            atoms.set_masses(gas.mass)
            G_gas = IdealGasThermo(
                vib_energies=gas.freqs[gas.ncut:],
                geometry='linear' if gas.linear else 'nonlinear',
                potentialenergy=gas.potential_energy + gas.dE, atoms=atoms,
                symmetrynumber=gas.symm, spin=gas.spin).get_gibbs_energy(
                    T, pressure=1000 * _Nav * _k * T, verbose=False)
            A_ads = HarmonicThermo(
                vib_energies=ads.freqs,
                potentialenergy=ads.potential_energy + ads.dE
            ).get_helmholtz_energy(T, verbose=False)
            K = np.exp(-(A_ads - G_gas) / (kB * T))
            self.assertAlmostEqual(float(rxn.keq) / K, 1., places=10,
                                   msg=gas_name)

    def test_symmetry_number_divides_orientations(self):
        # sigma = (lattice orientations) / symm (Hermes et al. 2019, eqs.
        # 6-7): an end-to-end symmetric two-site species on a hexagonal
        # lattice has 6 / 2 = 3 orientations
        from ase.units import kB
        T = 548.
        S = {}
        for symm in (1, 2):
            sp = wgs.build_species()
            ts = sp['o-h-oh']
            ts.lattice = Lattice({sp['slab']: {sp['slab']: 6}})
            ts.symm = symm
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                S[symm] = ts.get_S(T)
        self.assertAlmostEqual(S[2] - (ts.S['elec'] + ts.S['vib']),
                               kB * np.log(3), places=12)
        self.assertAlmostEqual((S[2] - S[1]) / kB, -np.log(2), places=12)
        # a single-site species has no orientations counted, so symm > 1
        # only reduces its partition function
        ads = wgs.build_species()['co']
        ads.symm = 2
        with self.assertWarns(UserWarning):
            ads.get_S(T)

    def test_reference_pressure(self):
        # Gas(pref=1) is the ideal gas at 1 bar (CatMap's convention); rate
        # constants do not depend on the reference state
        from ase.thermochemistry import IdealGasThermo
        from ase.db import connect
        T = 548.

        def rate_constants(**gas_kw):
            # no lateral interactions or energy shifts: the default
            # (computed) BEP alpha depends on the gas reference state, and
            # it only matters through those terms
            sp = wgs.build_species()
            for s in sp.values():
                s.lateral, s.dE = 0., 0.
            for name in ('co_g', 'co2_g'):
                for key, val in gas_kw.items():
                    setattr(sp[name], key, val)
            rxns = [Reaction(sp['co_g'], sp['co'], method='STICK'),
                    Reaction(sp['co_g'], sp['co'], method='EQUIL'),
                    Reaction(sp['o'] + sp['co'], sp['co2_g'],
                             ts=sp['o-co'])]
            k = []
            for rxn in rxns:
                rxn.update(T=T, Asite=wgs.ASITE, L=0)
                for x in (rxn.kfor, rxn.krev):
                    k.append(float(sympy.sympify(x).subs(
                        {s: 0 for s in sympy.sympify(x).free_symbols})))
            return sp, np.array(k)

        sp_m, k_m = rate_constants()
        sp_b, k_b = rate_constants(pref=1.)
        _, k_2 = rate_constants(rho0=2.)
        np.testing.assert_allclose(k_b, k_m, rtol=1e-12)
        np.testing.assert_allclose(k_2, k_m, rtol=1e-12)

        gas = sp_b['co_g']
        self.assertAlmostEqual(gas.get_reference_state(), bar_to_molar(1., T),
                               places=14)
        atoms = gas.atoms.copy()
        atoms.pbc = False
        atoms.set_masses(gas.mass)
        G_ase = IdealGasThermo(
            vib_energies=gas.freqs[gas.ncut:], geometry='linear',
            potentialenergy=gas.potential_energy + gas.dE, atoms=atoms,
            symmetrynumber=gas.symm, spin=gas.spin).get_gibbs_energy(
                T, pressure=1e5, verbose=False)
        self.assertAlmostEqual(gas.get_G(T), G_ase, places=10)

        with self.assertRaises(ValueError):
            Gas(gas.atoms, 'x', gas.freqs, rhoref=2., pref=1.)
        with tempfile.TemporaryDirectory() as tmp:
            db = connect(os.path.join(tmp, 'species.json'))
            gas.save_to_db(db)
            read = read_from_db(db)['co_g']
        self.assertEqual(read.pref, 1.)
        read.eref = gas.eref  # not stored in the db
        self.assertAlmostEqual(read.get_G(T), gas.get_G(T), places=12)

    def test_energy_reference_from_atoms(self):
        h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
        h2.calc = SinglePointCalculator(h2, energy=-6.8)
        self.assertAlmostEqual(EnergyReference([h2])['H'], -3.4)


class ReactionOptionsTest(unittest.TestCase):
    """clip, alpha and explicit_ts (CatMap-style conventions)."""

    T = 548.

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter('ignore')

    def setUp(self):
        self.sp = wgs.build_species()
        self.kT = kB * self.T

    def _update(self, rxn):
        rxn.update(T=self.T, Asite=wgs.ASITE, L=0)
        return rxn

    def _at(self, expr, **cov):
        # value of expr at the given coverages (others 0)
        expr = sympy.sympify(expr)
        subs = {s: cov.get(s.name, 0.) for s in expr.free_symbols}
        return float(expr.subs(subs))

    def test_invalid_options(self):
        sp = self.sp
        with self.assertRaisesRegex(ValueError, 'zero_coverage'):
            Reaction(sp['co'] + sp['o'], sp['co2_g'], ts=sp['o-co'],
                     dground=True)
        for kw in ({'clip': 'always'}, {'alpha': 1.5},
                   {'alpha': 0.5, 'explicit_ts': True}):
            with self.assertRaises(ValueError):
                Reaction(sp['co'] + sp['o'], sp['co2_g'], ts=sp['o-co'], **kw)
        for kw in ({'alpha': 0.5}, {'explicit_ts': True},
                   {'clip': 'zero_coverage'}):
            with self.assertRaises(ValueError):
                Reaction(sp['co_g'], sp['co'], **kw)
        with self.assertRaises(ValueError):
            Reaction(sp['co_g'], sp['co'], method='DIEQUIL', clip='coverage')

    def test_clip_coverage_equil_and_stick(self):
        # CatMap: barrierless kfor = kT/h exp(-max(0, dG)/kT), and the
        # non-activated (collision theory) prefactor times the same factor;
        # at high CO coverage, CO adsorption becomes endergonic
        sp = self.sp
        sp['co_g'].pref = 1.
        equil = self._update(Reaction(sp['co_g'], sp['co'], method='EQUIL',
                                      clip='coverage'))
        stick = self._update(Reaction(sp['co_g'], sp['co'], method='STICK',
                                      clip='coverage'))
        stick0 = self._update(Reaction(sp['co_g'], sp['co'], method='STICK'))
        ref = sp['co_g'].get_reference_state()
        signs = set()
        for theta in (0.1, 0.8):
            dG = self._at(equil.dG, co=theta)
            signs.add(dG > 0)
            factor = np.exp(-max(dG, 0.) / self.kT)
            k = _k * self.T / _hplanck * factor / ref
            self.assertAlmostEqual(self._at(equil.kfor, co=theta) / k, 1.,
                                   places=12)
            self.assertAlmostEqual(
                self._at(stick.kfor, co=theta)
                / (self._at(stick0.kfor, co=theta) * factor), 1., places=12)
            for rxn in (equil, stick):
                self.assertAlmostEqual(
                    self._at(rxn.kfor / rxn.krev, co=theta)
                    / self._at(rxn.keq, co=theta), 1., places=12)
        self.assertEqual(signs, {True, False})

    def test_clip_coverage_tst(self):
        # forward barrier max(dG_act, dG, 0), with an explicit transition
        # state lying below the final state
        sp = self.sp
        ts = sp['o-co']
        ts.dE = -2.
        rxn = Reaction(sp['co'] + sp['o'], sp['co2_g'], ts=ts,
                       explicit_ts=True, clip='coverage')
        self._update(rxn)
        for theta in (0., 0.5):
            dG_act = self._at(rxn.dG_act, co=theta)
            dG = self._at(rxn.dG, co=theta)
            barrier = max(dG_act, dG, 0.)
            self.assertGreater(barrier, dG_act)
            k = _k * self.T / _hplanck * np.exp(-barrier / self.kT)
            self.assertAlmostEqual(self._at(rxn.kfor, co=theta) / k, 1.,
                                   places=12)
        # without clip the negative barrier is an error
        with self.assertRaises(RuntimeError):
            self._update(Reaction(sp['co'] + sp['o'], sp['co2_g'], ts=ts,
                                  explicit_ts=True))

    def test_clip_zero_coverage(self):
        # the former dground: chosen at zero coverage, then kept. Shifts of
        # the TS itself are caught by the raw-barrier check; a reactant
        # shifted up by 3 eV, with a TS following the products, pushes the
        # barrier below 0
        sp = self.sp
        sp['co'].dE += 3.
        rxn = Reaction(sp['co'] + sp['o'], sp['co2_g'], ts=sp['o-co'],
                       alpha=1.)
        self.assertLess(self._at(self._update(rxn).dG_act), 0.)
        rxn = self._update(Reaction(sp['co'] + sp['o'], sp['co2_g'],
                                    ts=sp['o-co'], alpha=1.,
                                    clip='zero_coverage'))
        self.assertEqual(rxn.dG_act, 0.)

    def test_alpha_and_explicit_ts(self):
        # G_TS = G_TS,own + (1 - alpha) dE_reactants + alpha dE_products,
        # dE = lateral + energy shift; explicit_ts: G_TS,own only
        sp = self.sp
        ts = sp['ho-h']
        ts.lateral = 0.3 * sp['co'].symbol
        react, prod = sp['h2o'], sp['oh'] + sp['h']
        dEr = react.lateral + react.dE
        dEp = sum(s.lateral + s.dE for s in prod)
        base = Reaction(react, prod, ts=ts, alpha=0.)
        self._update(base)
        for kw, shift in [({'alpha': 0.}, 0.), ({'alpha': 1.}, dEp - dEr),
                          ({'alpha': 0.3}, 0.3 * (dEp - dEr)),
                          ({'explicit_ts': True}, -dEr)]:
            rxn = self._update(Reaction(react, prod, ts=ts, **kw))
            for theta in (0., 0.4):
                self.assertAlmostEqual(
                    self._at(rxn.dG_act - base.dG_act, co=theta),
                    self._at(shift, co=theta), places=12, msg=kw)
        # the default computes alpha, between 0 and 1
        rxn = self._update(Reaction(react, prod, ts=ts))
        self.assertTrue(0. < float(rxn.alpha) < 1.)


class FirstOrderLateralTest(unittest.TestCase):
    """micki.lateral.first_order against CatMap's response functions."""

    # catmap.functions.smooth_piecewise_linear(theta, slope=1.3,
    # cutoff=0.25, smoothing=s)[0] from CatMap 0.3.1, for s = 0.05 and 0
    CATMAP_F = {0.1: (0.0, 0.0),
                0.22: (0.01181818181818181, 0.0),
                0.25: (0.06499999999999999, 0.0),
                0.28: (0.1485714285714287, 0.1392857142857144),
                0.6: (0.7583333333333333, 0.7583333333333333)}

    def test_response_functions(self):
        from micki.lateral import _response
        theta = sympy.Symbol('theta')
        smooth = _response(theta, 'smooth_piecewise_linear', 1.3, 0.25, 0.05)
        piecewise = _response(theta, 'piecewise_linear', 1.3, 0.25, 0.05)
        for x, (f_smooth, f_piecewise) in self.CATMAP_F.items():
            self.assertAlmostEqual(float(smooth.subs(theta, x)), f_smooth,
                                   places=14)
            self.assertAlmostEqual(float(piecewise.subs(theta, x)),
                                   f_piecewise, places=14)
        self.assertEqual(float(_response(theta, 'linear', 1.3, 0.25, 0.05)),
                         1.3)
        with self.assertRaises(ValueError):
            _response(theta, 'cubic', 1., 0.25, 0.05)

    def test_first_order(self):
        from micki.lateral import first_order
        sp = wgs.build_species()
        ads = [sp[n] for n in ('co', 'h2o', 'oh', 'o', 'h', 'cooh')]
        co, o, ts = sp['co'], sp['o'], sp['o-co']
        eps = {co: {co: 1.5, o: 1.1}, o: {co: 1.1}, ts: {co: 0.7}}
        first_order(ads, eps)
        self.assertEqual(sympy.simplify(co.lateral - 1.5 * co.symbol
                                        - 1.1 * o.symbol), 0)
        first_order(ads, eps, response='piecewise_linear', cutoff=0.3)
        cov = {'co': 0.3, 'o': 0.05, 'h': 0.1}
        F = (sum(cov.values()) - 0.3) / sum(cov.values())
        subs = {s: cov.get(s.name, 0.) for s in co.lateral.free_symbols}
        self.assertAlmostEqual(float(co.lateral.subs(subs)),
                               F * (1.5 * 0.3 + 1.1 * 0.05), places=14)
        # below the cutoff, no interaction
        subs = {s: 0. for s in ts.lateral.free_symbols}
        subs[co.symbol] = 0.2
        self.assertEqual(float(ts.lateral.subs(subs)), 0.)
        with self.assertRaises(ValueError):
            first_order(ads, {co: {ts: 1.}})
        with self.assertRaises(ValueError):
            first_order(ads[:1], {co: {o: 1.}})

    def test_piecewise_steady_state(self):
        # a CSTR steady state with a piecewise-linear response, with the
        # complex-step Jacobian and IDA's difference quotients
        from micki.lateral import first_order
        sp = wgs.build_species()
        ads = [sp[n] for n in ('co', 'h2o', 'oh', 'o', 'h', 'cooh')]
        eps = {}
        for i in ads:
            terms = sympy.sympify(i.lateral).as_coefficients_dict()
            eps[i] = {j: float(terms[j.symbol]) for j in ads
                      if terms.get(j.symbol)}
        first_order(ads, eps, response='piecewise_linear')
        rxns = wgs.build_reactions(sp)
        T, p_co, p_h2o, p_co2, p_h2, flow = wgs.CONDITIONS[6]
        U0 = {'co_g': bar_to_molar(1.01325 * p_co, T),
              'h2o_g': bar_to_molar(1.01325 * p_h2o, T),
              'co2_g': 0., 'h2_g': 0.}
        U0.update(wgs.COVERAGE0)
        results = []
        for analytic_jac in (False, True):
            model = Model(T, wgs.ASITE, reactor='CSTR',
                          analytic_jac=analytic_jac)
            model.lattice = {sp['slab']: {sp['slab']: 6}}
            model.add_reactions(rxns)
            model.set_fixed(['co_g', 'h2o_g', 'h2_g', 'co2_g'])
            model.set_initial_conditions(U0)
            _, U, r = model.find_steady_state()
            results.append((float(U['co']), float(r['co_ads'])))
        np.testing.assert_allclose(results[1], results[0], rtol=1e-6)
        # the response lowers the CO repulsion below theta_tot = 1
        self.assertGreater(results[0][0], 0.47)


class ConventionsTest(unittest.TestCase):
    """micki.conventions('catmap') changes defaults, nothing else."""

    T = 548.

    def setUp(self):
        warnings.simplefilter('ignore')

    def _o_co(self, sp, **kw):
        return Reaction(sp['o'] + sp['co'], sp['co2_g'], ts=sp['o-co'], **kw)

    def test_default_unchanged(self):
        self.assertEqual(micki.get_conventions(), 'micki')
        sp = wgs.build_species()
        self.assertIsNone(sp['co_g'].pref)
        self.assertEqual(sorted(sp['co_g'].mass), [masses['C'], masses['O']])
        rxn = self._o_co(sp)
        self.assertEqual((rxn.clip, rxn.alpha_fixed, rxn.ts_follows_dE),
                         (None, None, True))

    def test_catmap_defaults(self):
        from ase.data import atomic_masses, atomic_numbers
        with micki.conventions('catmap'):
            sp = wgs.build_species()
            rxn = self._o_co(sp)
            explicit = self._o_co(sp, explicit_ts=True)
            overridden = self._o_co(sp, clip=None, alpha=None)
            stick = Reaction(sp['co_g'], sp['co'], method='STICK')
            diequil = Reaction(2 * sp['oh'], sp['o'] + sp['h2o'],
                               method='DIEQUIL')
            gas = Gas(sp['co_g'].atoms, 'x', sp['co_g'].freqs, rhoref=1.)
        self.assertEqual(micki.get_conventions(), 'micki')
        self.assertEqual(sp['co_g'].pref, 1.)
        self.assertEqual(sorted(sp['co_g'].mass),
                         [atomic_masses[atomic_numbers[s]] for s in 'CO'])
        self.assertEqual((rxn.clip, rxn.alpha_fixed, rxn.ts_follows_dE),
                         ('coverage', 0.5, False))
        self.assertEqual((explicit.alpha_fixed, explicit.explicit_ts),
                         (None, True))
        self.assertEqual((overridden.clip, overridden.alpha_fixed),
                         (None, None))
        self.assertEqual(stick.clip, 'coverage')
        self.assertIsNone(diequil.clip)
        self.assertIsNone(gas.pref)
        # copies keep their conventions
        self.assertEqual(sp['co_g'].copy().pref, 1.)
        self.assertEqual(sp['co'].copy().conventions, 'catmap')

    def test_ts_does_not_follow_dE(self):
        # under CatMap's conventions a reactant's dE moves only the
        # reactant: the forward barrier drops by the full shift
        barriers = {}
        for name in ('micki', 'catmap'):
            with micki.conventions(name):
                sp = wgs.build_species()
                rxn = self._o_co(sp, alpha=0.5, clip=None)
            for shift in (0., 0.05):
                sp['co'].dE += shift
                rxn.update(T=self.T, Asite=wgs.ASITE, L=0, force=True)
                barriers[name, shift] = float(sympy.sympify(rxn.dG_act).subs(
                    {s: 0 for s in sympy.sympify(rxn.dG_act).free_symbols}))
        self.assertAlmostEqual(barriers['micki', 0.05]
                               - barriers['micki', 0.], -0.025, places=10)
        self.assertAlmostEqual(barriers['catmap', 0.05]
                               - barriers['catmap', 0.], -0.05, places=10)

    def test_model_checks(self):
        sp = wgs.build_species()
        with micki.conventions('catmap'):
            sp_c = wgs.build_species()
        model = Model(self.T, wgs.ASITE)
        with self.assertRaisesRegex(ValueError, 'conventions'):
            model.add_reactions({'a': self._o_co(sp), 'b': self._o_co(sp_c)})
        with micki.conventions('catmap'):
            rxn = self._o_co(sp_c)
        with self.assertRaisesRegex(ValueError, 'conventions'):
            Model(self.T, wgs.ASITE).add_reactions(
                {'a': Reaction(sp['o'] + sp['co'], sp['co2_g'],
                               ts=sp_c['o-co'])})
        model = Model(self.T, wgs.ASITE)
        model.add_reactions({'a': rxn})
        with self.assertWarnsRegex(RuntimeWarning, 'lattice'):
            model.lattice = {sp_c['slab']: {sp_c['slab']: 6}}

    def test_invalid_name(self):
        with self.assertRaises(ValueError):
            micki.set_conventions('cantera')
        with self.assertRaises(ValueError):
            with micki.conventions('cantera'):
                pass


def _with_energy(atoms, energy=0.):
    atoms.calc = SinglePointCalculator(atoms, energy=energy)
    return atoms


class ReactionStringTest(unittest.TestCase):
    """Reaction.from_string, reactions_from_strings, atom balance."""

    T = 548.

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter('ignore')
        cls.sp = wgs.build_species()

    def _constants(self, rxn):
        rxn.update(T=self.T, Asite=wgs.ASITE, L=0)
        return [float(sympy.sympify(k).subs(
            {s: 0.1 for s in sympy.sympify(k).free_symbols}))
            for k in (rxn.keq, rxn.kfor, rxn.krev)]

    def test_same_as_operators(self):
        sp = self.sp
        pairs = [
            (Reaction.from_string('co + o <-> o-co -> co2_g', sp),
             Reaction(sp['o'] + sp['co'], sp['co2_g'], ts=sp['o-co'])),
            (Reaction.from_string('h2_g -> 2 h', sp, method='STICK'),
             Reaction(sp['h2_g'], 2 * sp['h'], method='STICK')),
            (Reaction.from_string('h2_g + 2*slab -> 2h', sp, method='STICK'),
             Reaction(sp['h2_g'], 2 * sp['h'], method='STICK')),
            (Reaction.from_string('cooh + slab <-> oco-h + slab -> co2_g + h',
                                  sp),
             Reaction(sp['cooh'] + sp['slab'], sp['co2_g'] + sp['h'],
                      ts=sp['oco-h'])),
            (Reaction.from_string('2 oh -> o + h2o', list(sp.values()),
                                  method='DIEQUIL'),
             Reaction(2 * sp['oh'], sp['o'] + sp['h2o'], method='DIEQUIL')),
        ]
        for new, old in pairs:
            self.assertEqual(sorted(s.label for s in new.reactants),
                             sorted(s.label for s in old.reactants))
            self.assertEqual(sorted(s.label for s in new.products),
                             sorted(s.label for s in old.products))
            self.assertEqual(new.method, old.method)
            np.testing.assert_allclose(self._constants(new),
                                       self._constants(old), rtol=1e-12)
        # the transition-state part ignores empty sites
        self.assertEqual([s.label for s in pairs[3][0].ts], ['oco-h'])

    def test_catmap_names(self):
        # CatMap's syntax with a dict mapping its names to micki species
        sp = self.sp
        names = {'CO_g': sp['co_g'], 'H2_g': sp['h2_g'], '*_s': sp['slab'],
                 'CO_s': sp['co'], 'O_s': sp['o'], 'H_s': sp['h'],
                 'O-CO_s': sp['o-co'], 'CO2_g': sp['co2_g']}
        for expression, old in [
                ('CO_g + *_s -> CO_s',
                 Reaction(sp['co_g'], sp['co'])),
                ('H2_g + 2*_s -> 2H_s',
                 Reaction(sp['h2_g'], 2 * sp['h'])),
                ('CO_s + O_s <-> O-CO_s + *_s -> CO2_g + 2*_s',
                 Reaction(sp['co'] + sp['o'], sp['co2_g'], ts=sp['o-co']))]:
            new = Reaction.from_string(expression, names)
            self.assertEqual(sorted(s.label for s in new.reactants),
                             sorted(s.label for s in old.reactants))
            self.assertEqual(sorted(s.label for s in new.products),
                             sorted(s.label for s in old.products))
            np.testing.assert_allclose(self._constants(new),
                                       self._constants(old), rtol=1e-12)

    def test_star(self):
        # '*' is the only empty-site species, whatever its label
        sp = self.sp
        for expression, old in [
                ('co_g + * -> co', Reaction(sp['co_g'], sp['co'])),
                ('h2_g + 2* -> 2 h', Reaction(sp['h2_g'], 2 * sp['h'])),
                ('cooh + * <-> oco-h + * -> co2_g + h',
                 Reaction(sp['cooh'] + sp['slab'], sp['co2_g'] + sp['h'],
                          ts=sp['oco-h']))]:
            new = Reaction.from_string(expression, sp)
            self.assertEqual(sorted(s.label for s in new.reactants),
                             sorted(s.label for s in old.reactants))
            self.assertEqual(sorted(s.label for s in new.products),
                             sorted(s.label for s in old.products))
            if old.ts is not None:
                self.assertEqual([s.label for s in new.ts], ['oco-h'])
        # ambiguous with two site types
        from micki import Adsorbate
        top = Adsorbate(_with_energy(Atoms()), 'top', [])
        hollow = Adsorbate(_with_energy(Atoms()), 'hollow', [])
        h = Adsorbate(_with_energy(Atoms('H')), 'h', [0.1, 0.1, 0.1],
                      sites=[hollow])
        co = Adsorbate(_with_energy(Atoms('CO')), 'co', [0.2, 0.1, 0.1],
                       sites=[top])
        with self.assertRaisesRegex(ValueError, 'hollow, top'):
            Reaction.from_string('h2_g + 2* -> 2 h',
                                 [sp['h2_g'], top, hollow, h, co])
        Reaction.from_string('h2_g + 2 hollow -> 2 h',
                             [sp['h2_g'], top, hollow, h, co])

    def test_empty_sites_by_reference(self):
        # an adsorbate created without sites is not mistaken for an empty
        # site (only species listed in other species' sites are)
        from micki import Adsorbate
        co_g = self.sp['co_g']
        Reaction(co_g, Adsorbate(_with_energy(Atoms('CO')), 'bare_co',
                                 [0.2, 0.1, 0.1]))
        with self.assertRaisesRegex(ValueError, 'conserve atoms'):
            Reaction(co_g, Adsorbate(_with_energy(Atoms('O')), 'bare_o',
                                     [0.1]))

    def test_errors(self):
        sp = self.sp
        for expression in ('co + o', 'co -> o -> co2_g -> o',
                           'co + -> o', 'cO -> co_g', '0 co -> co_g',
                           'co + o <-> slab -> co2_g', 'c o -> co_g'):
            with self.assertRaises(ValueError, msg=expression):
                Reaction.from_string(expression, sp)
        with self.assertRaisesRegex(ValueError, 'twice'):
            Reaction.from_string('co + o <-> o-co -> co2_g', sp,
                                 ts=sp['o-co'])

    def test_balance(self):
        sp = self.sp
        with self.assertRaisesRegex(ValueError, 'conserve atoms'):
            Reaction(sp['co_g'], sp['o'])
        with self.assertRaisesRegex(ValueError, 'transition state'):
            Reaction(sp['co'] + sp['o'], sp['co2_g'], ts=sp['co-oh'])
        with self.assertRaisesRegex(ValueError, 'conserve atoms'):
            Reaction.from_string('co_g -> o', sp)
        Reaction(sp['co_g'], sp['o'], check_balance=False)

    def test_reactions_from_strings(self):
        sp = self.sp
        rxns = micki.reactions_from_strings(sp, {
            'co_ads': ('co_g -> co', {'method': 'STICK'}),
            'co_ox': 'co + o <-> o-co -> co2_g',
        })
        self.assertEqual(list(rxns), ['co_ads', 'co_ox'])
        self.assertEqual(rxns['co_ads'].method, 'STICK')
        self.assertEqual(rxns['co_ox'].method, 'TST')
        with self.assertRaisesRegex(ValueError, 'Reaction bad'):
            micki.reactions_from_strings(sp, {'bad': 'co -> nothing'})


class LatticeTest(unittest.TestCase):

    def test_update_site_names(self):
        sp = wgs.build_species()
        a, b = sp['slab'], sp['co']
        by_name = Lattice({'a': {'a': 2, 'b': 4}, 'b': {'a': 4, 'b': 2}})
        by_name.update_site_names({'a': a, 'b': b})
        direct = Lattice({a: {a: 2, b: 4}, b: {a: 4, b: 2}})
        self.assertFalse(by_name.string_names)
        self.assertEqual(by_name.neighborlist, direct.neighborlist)
        self.assertEqual(by_name.ratio, direct.ratio)


class VdWRadiusTest(unittest.TestCase):

    def test_single_atom(self):
        from ase.data import vdw_radii
        r = calculate_avg_vdw_radius(Atoms('Ar', positions=[[0, 0, 0]]), 101)
        self.assertAlmostEqual(r, vdw_radii[18])

    def test_diatomic_against_ray_marching(self):
        from ase.data import vdw_radii
        atoms = Atoms('N2', positions=[[0, 0, 0], [0, 0, 1.1]])
        npoints = 201
        r = calculate_avg_vdw_radius(atoms, npoints)
        # independent estimate: march along each ray and record the last
        # point inside a vdW sphere
        pos = atoms.get_positions() - atoms.get_center_of_mass()
        R = vdw_radii[7]
        offset = 2 / npoints
        increment = np.pi * (3 - np.sqrt(5))
        steps = np.linspace(0, 4, 40001)
        total = 0.
        for i in range(npoints):
            y = i * offset - 1 + offset / 2
            phi = ((i + 1) % npoints) * increment
            vec = np.array([np.cos(phi) * np.sqrt(1 - y**2), y,
                            np.sin(phi) * np.sqrt(1 - y**2)])
            pts = steps[:, None] * vec
            inside = np.zeros(len(steps), dtype=bool)
            for p in pos:
                inside |= np.linalg.norm(pts - p, axis=1) <= R
            total += steps[np.nonzero(inside)[0].max()]
        self.assertAlmostEqual(r, total / npoints, places=3)


if __name__ == '__main__':
    unittest.main()
