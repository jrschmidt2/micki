"""This module contains object definitions of species
and collections of species"""

import copy
import warnings
import numpy as np

from sympy import Symbol

from ase import Atoms
from ase.db import connect
from ase.db.core import Database
from ase.db.row import AtomsRow
from ase.units import mol, _hplanck, m, kg, _k, kB

from micki.masses import masses
from micki.io import parse_vasp_out
from micki.utils import calculate_avg_vdw_radius


class _Thermo:
    """Generic thermodynamics object

    This is the base object that all reactant objects inherit from.
    It initializes many parameters and provides methods for calculating
    the partition function from translation, rotation, and vibration."""

    def __init__(self):
        self.T = None

        self.mode = ['tot', 'trans', 'trans2D', 'rot', 'vib', 'elec']

        self.q = dict.fromkeys(self.mode)
        self.S = dict.fromkeys(self.mode)
        self.E = dict.fromkeys(self.mode)
        self.H = None

        self.scale = {'E': dict.fromkeys(self.mode, 1.0),
                      'S': dict.fromkeys(self.mode, 1.0),
                      'H': 1.0}
        self.scale_old = copy.deepcopy(self.scale)

        self.atoms = None
        self.metal = None
        self.eref = None
        self.potential_energy = 0.
        self.symm = 1
        self.spin = 0.
        self.ts = False
        self.label = None
        self.lateral = 0.
        self.dE = 0.
        self.sites = []
        self.lattice = None
        self.D = None
        self.Sliq = None
        self.rho0 = 1.
        self.freqs = []

    def set_atoms(self, atoms):
        if atoms is None:
            self._atoms = atoms
            return
        elif isinstance(atoms, AtomsRow):
            self._atoms = atoms.toatoms()
            self.freqs = atoms.data.get('freqs')
        elif isinstance(atoms, Atoms):
            self._atoms = atoms
        elif isinstance(atoms, str):
            # TODO: make this more robust (catch/handle errors)
            a, f = parse_vasp_out(atoms)
            self._atoms = a
            self.freqs = f
        else:
            raise ValueError("Unrecognized atoms object!")
        self.mass = [masses[atom.symbol] for atom in self.atoms]
        self.atoms.set_masses(self.mass)
        self.update_potential_energy()

    def get_atoms(self):
        return self._atoms

    atoms = property(get_atoms, set_atoms)

    def set_reference(self, reference):
        self._eref = reference
        self.update_potential_energy()

    def get_reference(self):
        return self._eref

    eref = property(get_reference, set_reference)

    def update_potential_energy(self):
        if self.atoms is None or len(self.atoms) == 0:
            self.potential_energy = 0.
        else:
            self.potential_energy = self.atoms.get_potential_energy()
        if self.eref is not None:
            for element in self.atoms.get_chemical_symbols():
                self.potential_energy -= self.eref[element]
    
    def set_sites(self, sites):
        if isinstance(sites, list):
            self._sites = sites
        elif isinstance(sites, Adsorbate):
            self._sites = [sites]
        else:
            raise ValueError("Invalid format for adsorption sites")

    def get_sites(self):
        return self._sites

    sites = property(get_sites, set_sites)

    def set_freqs(self, freqs):
        if freqs is not None:
            self._freqs = np.array(freqs)

    def get_freqs(self):
        return self._freqs

    freqs = property(get_freqs, set_freqs)

    def set_label(self, label):
        self._label = label
        if label is None:
            self._symbol = None
        else:
            self._symbol = Symbol(label)

    def get_label(self):
        return self._label

    label = property(get_label, set_label)

    def get_symbol(self):
        return self._symbol

    symbol = property(get_symbol, None)

    def update(self, T=None, force=False):
        """Updates the object's thermodynamic properties"""
        if not self.is_update_needed(T) and not force:
            return

        if T is None:
            T = self.T

        self.T = T
        self._calc_q(T)
        self.scale_old = copy.deepcopy(self.scale)

    def is_update_needed(self, T):
        if self.q['tot'] is None:
            return True
        if T is not None and T != self.T:
            return True
        if self.scale != self.scale_old:
            return True
        return False

    def get_H(self, T=None):
        self.update(T)
        return (self.H + self.lateral) * self.scale['H']

    def get_S(self, T=None):
        self.update(T)
        return self.S['tot'] * self.scale['S']['tot']

    def get_G(self, T=None):
        self.update(T)
        T = self.T
        return self.get_H(T) - T * self.get_S(T)

    def get_E(self, T=None):
        self.update(T)
        return (self.E['tot'] + self.lateral) * self.scale['E']['tot']

    def get_q(self, T=None):
        self.update(T)
        return self.q['tot']

    def get_reference_state(self):
        raise NotImplementedError

    def save_to_db(self, db):
        if isinstance(db, str):
            db = connect(db)
        elif not isinstance(db, Database):
            raise ValueError("Must pass active ASE DB connection, or name of ASE DB file!")

        data = {'freqs': self.freqs,
                'ts': self.ts,
                'symm': self.symm,
                'spin': self.spin,
                'D': self.D,
                'S': self.Sliq,
                'rhoref': self.rho0,
                'sites': [site.label for site in self.sites],
                'dE': self.dE}

        if isinstance(self, Adsorbate):
            data['thermo'] = 'Adsorbate'
        elif isinstance(self, Gas):
            data['thermo'] = 'Gas'
        elif isinstance(self, Liquid):
            data['thermo'] = 'Liquid'
        else:
            raise ValueError("Unknown Thermo object type {}".format(type(self)))

        if self.lattice:
            warnings.warn('Lattice cannot be stored in a db! You must recreate '
                          'the lattice when you re-use this species.',
                          RuntimeWarning, stacklevel=2)

        if self.eref:
            warnings.warn('Energy reference cannot be stored in a db! You must '
                          'recreate the energy reference when you re-use this '
                          'species.', RuntimeWarning, stacklevel=2)

        if self.lateral != 0.:
            warnings.warn('Coverage dependence cannot be stored in a db! You '
                          'must recreate the coverage dependence when you '
                          're-use this species.', RuntimeWarning, stacklevel=2)

        db.write(self.atoms, name=self.label, data=data)

    def _calc_q(self, T):
        raise NotImplementedError

    def _calc_qtrans2D(self, T, A):
        mtot = sum(self.mass) / kg
        self.q['trans2D'] = 2 * np.pi * mtot * _k * T / _hplanck**2 * A
        self.E['trans2D'] = kB * T * self.scale['E']['trans2D']
        self.S['trans2D'] = kB * (2. + np.log(self.q['trans2D'])) * \
            self.scale['S']['trans2D']

    def _calc_qtrans(self, T):
        mtot = sum(self.mass) / kg
        self.q['trans'] = 0.001*(2*np.pi*mtot*_k*T/_hplanck**2)**(3./2.) \
            / (mol * self.rho0)
        self.E['trans'] = 3. * kB * T / 2. * self.scale['E']['trans']
        self.S['trans'] = kB * (5./2. + np.log(self.q['trans'])) * \
            self.scale['S']['trans']

    def _calc_qrot(self, T):
        com = self.atoms.get_center_of_mass()
        if self.linear:
            I = 0
            for atom in self.atoms:
                I += atom.mass * np.linalg.norm(atom.position - com)**2
            I /= (kg * m**2)
            self.q['rot'] = 8*np.pi**2*I*_k*T/(_hplanck**2*self.symm)
            self.E['rot'] = kB * T * self.scale['E']['rot']
            self.S['rot'] = kB * (1. + np.log(self.q['rot'])) * \
                self.scale['S']['rot']
        else:
            I = self.atoms.get_moments_of_inertia() / (kg * m**2)
            thetarot = _hplanck**2 / (8 * np.pi**2 * I * _k)
            self.q['rot'] = np.sqrt(np.pi*T**3/np.prod(thetarot))/self.symm
            self.E['rot'] = 3. * kB * T / 2. * self.scale['E']['rot']
            self.S['rot'] = kB * (3./2. + np.log(self.q['rot'])) * \
                self.scale['S']['rot']

    def _calc_qvib(self, T, ncut=0):
        thetavib = self.freqs[ncut:] / kB
        self.q['vib'] = np.prod(np.exp(-thetavib/(2. * T)) /
                                (1. - np.exp(-thetavib/T)))
        self.E['vib'] = kB * sum(thetavib *
                                 (1./2. + 1./(np.exp(thetavib/T) - 1.))) * \
            self.scale['E']['vib']
        self.S['vib'] = kB * sum((thetavib/T)/(np.exp(thetavib/T) - 1.) -
                                 np.log(1. - np.exp(-thetavib/T))) * \
            self.scale['S']['vib']

    def _calc_qelec(self, T):
        self.E['elec'] = self.potential_energy + self.dE
        self.E['elec'] *= self.scale['E']['elec']
        self.S['elec'] = kB * np.log(2. * self.spin + 1.) * \
            self.scale['S']['elec']

    def _is_linear(self):
        pos = self.atoms.get_positions()
        vecs = pos[1:] - pos[0]
        for vec in vecs[1:]:
            if np.linalg.norm(np.cross(vecs[0], vec)) > 1e-8:
                return False
        return True

    def copy(self):
        raise NotImplementedError

    def __repr__(self):
        if self.label is not None:
            return self.label
        else:
            return self.atoms.get_chemical_formula()

    def __add__(self, other):
        return _Reactants([self, other])

    def __iadd__(self, other):
        raise NotImplementedError

    def __mul__(self, factor):
        if not isinstance(factor, int) or factor <= 0:
            raise ValueError('Can only multiply a species by a positive '
                             'integer')
        return _Reactants([self for i in range(factor)])

    def __rmul__(self, factor):
        return self.__mul__(factor)


class _Fluid(_Thermo):
    """Master object for both liquids and gasses"""
    def __init__(self, atoms, label, freqs=None, symm=1, spin=0.,
                 eref=None, rhoref=1., dE=0.):
        _Thermo.__init__(self)
        self.atoms = atoms
        self.freqs = freqs
        self.label = label
        self.symm = symm
        self.spin = spin
        self.eref = eref
        self.linear = self._is_linear()
        self.ncut = 6 - self.linear + self.ts
        self.rho0 = rhoref
        self.dE = dE
        self._R = None
        if not np.all(self.freqs[self.ncut:] > 0):
            raise ValueError("Extra imaginary frequencies found for {}!"
                             "".format(label))

    def get_reference_state(self):
        return self.rho0

    def copy(self, newlabel=None):
        label = self.label
        if newlabel is not None:
            label = newlabel
        return self.__class__(self.atoms, label, self.freqs,
                              self.symm, self.spin, self.eref,
                              self.rho0, self.dE)

    def _calc_q(self, T):
        self._calc_qelec(T)
        self._calc_qtrans(T)
        self._calc_qrot(T)
        self._calc_qvib(T, ncut=self.ncut)
        self.q['tot'] = self.q['trans'] * self.q['rot'] * self.q['vib']
        self.E['tot'] = self.E['elec'] + self.E['trans'] + self.E['rot'] + \
            self.E['vib']
        self.H = self.E['tot']
        self.S['tot'] = self.S['elec'] + self.S['trans'] + self.S['rot'] + \
            self.S['vib']

    def get_R(self):
        if self._R is None:
            self._R = calculate_avg_vdw_radius(self.atoms)
        return self._R
    
    R = property(get_R, None)


class Electron(_Thermo):
    def __init__(self, E, self_repulsion, label):
        _Thermo.__init__(self)
        self.atoms = Atoms()
        self.potential_energy = E
        self.label = label
        self.self_repulsion = self_repulsion
        self.lateral = self_repulsion * self.symbol

    def get_reference_state(self):
        return 1.

    def copy(self, newlabel=None):
        label = self.label
        if newlabel is not None:
            label = newlabel
        return self.__class__(self.potential_energy, self.self_repulsion,
                              label)

    def _calc_q(self, T):
        self._calc_qelec(T)
        if self.q['elec'] is None:
            self.q['elec'] = 1.
        self.q['tot'] = self.q['elec']
        self.E['tot'] = self.E['elec']
        self.H = self.E['tot']
        self.S['tot'] = self.S['elec']


class Gas(_Fluid):
    pass


class Liquid(_Fluid):
    def __init__(self, atoms, label, freqs=None, symm=1,
                 spin=0., eref=None, rhoref=1., S=None, D=None, dE=0.):
        _Fluid.__init__(self, atoms, label, freqs, symm, spin, eref,
                        rhoref, dE)
        self.Sliq = S
        self.D = D

    def copy(self, newlabel=None):
        label = self.label
        if newlabel is not None:
            label = newlabel
        return self.__class__(self.atoms, label, self.freqs,
                              self.symm, self.spin, self.eref,
                              self.rho0, self.Sliq, self.D, self.dE)


class Adsorbate(_Thermo):
    def __init__(self, atoms, label, freqs=None, ts=None,
                 spin=0., sites=None, lattice=None, eref=None, dE=0.,
                 symm=1):
        _Thermo.__init__(self)
        self.atoms = atoms
        self.freqs = freqs
        self.label = label
        self.ts = ts
        self.spin = spin
        self.sites = [] if sites is None else sites
        self.lattice = lattice
        self.eref = eref
        self.dE = dE
        self.symm = symm
        if not np.all(self.freqs[1 if ts else 0:] > 0):
            raise ValueError("Imaginary frequencies found for {}!"
                             "".format(label))

    def get_reference_state(self):
        return 1.

    def _calc_q(self, T):
        self._calc_qvib(T, ncut=1 if self.ts else 0)
        self._calc_qelec(T)
        self.q['tot'] = self.q['vib']
        self.E['tot'] = self.E['elec'] + self.E['vib']
        self.H = self.E['tot']
        self.S['tot'] = self.S['elec'] + self.S['vib']
        self.S['tot'] += kB * np.log(self.symm)
        if self.lattice is not None:
            self.S['tot'] += self.lattice.get_S_conf(self.sites)


    def copy(self, newlabel=None):
        label = self.label
        if newlabel is not None:
            label = newlabel
        return self.__class__(self.atoms, label, self.freqs,
                              self.ts, self.spin, self.sites,
                              self.lattice, self.eref, self.dE,
                              self.symm)


class _Reactants:
    def __init__(self, species):
        self.species = []
        self.elements = {}
        for i, other in enumerate(species):
            if isinstance(other, _Reactants):
                # If we're adding a _Reactants object to another
                # _Reactants object, just merge species and elements.
                self.species += other.species
                for key in other.elements:
                    if key in self.elements:
                        self.elements[key] += other.elements[key]
                    else:
                        self.elements[key] = other.elements[key]

            elif isinstance(other, _Thermo):
                # If we're adding a _Thermo object to a reactants
                # object, append the _Thermo to species and update
                # elements
                self.species.append(other)
                for symbol in other.atoms.get_chemical_symbols():
                    if symbol in self.elements:
                        self.elements[symbol] += 1
                    else:
                        self.elements[symbol] = 1

            else:
                raise NotImplementedError
        self.reference_state = 1.
        for species in self.species:
            self.reference_state *= species.get_reference_state()

    def get_H(self, T=None):
        H = 0.
        for species in self.species:
            H += species.get_H(T)
        return H

    def get_S(self, T=None):
        S = 0.
        for species in self.species:
            S += species.get_S(T)
        return S

    def get_G(self, T=None):
        G = 0.
        for species in self.species:
            G += species.get_G(T)
        return G

    def get_E(self, T=None):
        E = 0.
        for species in self.species:
            E += species.get_E(T)
        return E

    def get_q(self, T=None):
        q = 1.
        for species in self.species:
            q *= species.get_q(T)
        return q

    def get_reference_state(self):
        return self.reference_state

    def copy(self):
        return self.__class__(self.species)

    def get_mass(self):
        mtot = 0
        for species in self.species:
            mtot += species.atoms.get_masses().sum()
        return mtot

    def __iadd__(self, other):
        if isinstance(other, _Reactants):
            self.species += other.species
            for key in other.elements:
                if key in self.elements:
                    self.elements[key] += other.elements[key]
                else:
                    self.elements[key] = other.elements[key]

        elif isinstance(other, _Thermo):
            self.species.append(other)
            for symbol in other.atoms.get_chemical_symbols():
                if symbol in self.elements:
                    self.elements[symbol] += 1
                else:
                    self.elements[symbol] = 1
        else:
            raise NotImplementedError
        return self

    def __add__(self, other):
        return _Reactants([self, other])

    def __imul__(self, factor):
        if not isinstance(factor, int) or factor <= 0:
            raise ValueError('Can only multiply by a positive integer')
        self.species *= factor
        for key in self.elements:
            self.elements[key] *= factor
        return self

    def __mul__(self, factor):
        new = self.copy()
        new *= factor
        return new

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __repr__(self):
        return ' + '.join([species.__repr__() for species in self.species])

    def __getitem__(self, i):
        return self.species[i]

    def __len__(self):
        return len(self.species)
