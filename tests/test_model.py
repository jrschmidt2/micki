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
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wgs  # noqa: E402
from micki import (Model, Reaction, Liquid, Electron,  # noqa: E402
                   EnergyReference, Lattice)
from micki.db import read_from_db  # noqa: E402
from micki.utils import calculate_avg_vdw_radius  # noqa: E402


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

    def test_energy_reference_from_atoms(self):
        h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.74]])
        h2.calc = SinglePointCalculator(h2, energy=-6.8)
        self.assertAlmostEqual(EnergyReference([h2])['H'], -3.4)


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
