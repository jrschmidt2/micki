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

from ase.data import atomic_masses, atomic_numbers

from micki.conventions import get_conventions, conventions
from micki.masses import masses
from micki.io import parse_vasp_out
from micki.utils import calculate_avg_vdw_radius, bar_to_molar


class _Thermo:
    """Base class of all species (Gas, Liquid, Adsorbate, Electron).

    Computes the partition function and the thermodynamic functions of a
    species at temperature T from its electronic energy (atoms.calc
    energy, minus the energy reference eref, plus dE) and its vibrational
    frequencies (plus translation and rotation for fluids). Results are
    cached per temperature (update()).

    Common attributes:

    label : str
        Name of the species; also the name of its sympy Symbol (symbol),
        which stands for its concentration (fluids, M) or coverage
        (adsorbates) in rate expressions.
    dE : float
        Energy shift in eV added to the electronic energy, e.g. a
        correction or fitting parameter. Also used by thermodynamic rate
        control.
    lateral : float or sympy expression
        Coverage-dependent energy in eV (lateral interactions), in terms of
        the symbols of adsorbates, e.g. 0.3 * co.symbol. Added to E, H and
        G. The interaction matrix must be symmetric to be thermodynamically
        consistent (Hermes et al., J. Chem. Phys. 151, 014112 (2019)).
    scale : dict
        Multipliers of the energy, entropy and enthalpy contributions
        ('E', 'S': per mode, 'H'), for sensitivity analysis.
    conventions : str
        The micki.conventions in effect when the species was built.
    """

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

        # conventions in effect when the species was built (micki.conventions)
        self.conventions = get_conventions()
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
        self.sitefree = False
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
        if self.conventions == 'catmap':
            # standard atomic weights, as CatMap (via ASE) uses
            self.mass = [atomic_masses[atomic_numbers[atom.symbol]]
                         for atom in self.atoms]
        else:
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
        """Recompute the thermodynamic properties at T (if T, or the scale
        factors, changed since the last update, or if force)."""
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
        """Enthalpy (eV) at T, including lateral interactions (a sympy
        expression in the coverages if there are any)."""
        self.update(T)
        return (self.H + self.lateral) * self.scale['H']

    def get_S(self, T=None):
        """Entropy (eV/K) at T."""
        self.update(T)
        return self.S['tot'] * self.scale['S']['tot']

    def get_G(self, T=None):
        """Free energy G = H - TS (eV) at T: for fluids the chemical
        potential at the reference state, for adsorbates the Helmholtz
        energy."""
        self.update(T)
        T = self.T
        return self.get_H(T) - T * self.get_S(T)

    def get_E(self, T=None):
        """Internal energy (eV) at T, including zero-point energy and
        lateral interactions."""
        self.update(T)
        return (self.E['tot'] + self.lateral) * self.scale['E']['tot']

    def get_q(self, T=None):
        """Partition function at T (fluids: per molecule at the reference
        concentration)."""
        self.update(T)
        return self.q['tot']

    def get_reference_state(self):
        raise NotImplementedError

    def save_to_db(self, db):
        """Write the species to an ASE database (a file name or an open
        connection), to be read back with micki.db.read_from_db. Lattices,
        energy references and lateral interactions are not stored (a
        warning says so)."""
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
                # with a reference pressure, rho0 depends on T
                'rhoref': 1. if getattr(self, 'pref', None) else self.rho0,
                'pref': getattr(self, 'pref', None),
                'sites': [site.label for site in self.sites],
                'sitefree': self.sitefree,
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
    """Common base class of Gas and Liquid (ideal-gas translation, rigid
    rotor, harmonic vibrations); see Gas for the parameters."""
    def __init__(self, atoms, label, freqs=None, symm=1, spin=0.,
                 eref=None, rhoref=None, dE=0., pref=None):
        _Thermo.__init__(self)
        self.atoms = atoms
        self.freqs = freqs
        self.label = label
        self.symm = symm
        self.spin = spin
        self.eref = eref
        self.linear = self._is_linear()
        self.ncut = 6 - self.linear + self.ts
        if pref is not None and rhoref is not None:
            raise ValueError('Give either rhoref or pref, not both!')
        if pref is None and rhoref is None and isinstance(self, Gas) \
                and self.conventions == 'catmap':
            pref = 1.
        self.rho0 = 1. if rhoref is None else rhoref
        # reference pressure in bar; if given, the reference concentration
        # rho0 = pref / RT (in M) follows the temperature
        self.pref = pref
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
        with conventions(self.conventions):
            return self.__class__(self.atoms, label, self.freqs,
                                  self.symm, self.spin, self.eref,
                                  None if self.pref else self.rho0,
                                  self.dE, self.pref)

    def _calc_q(self, T):
        if self.pref is not None:
            self.rho0 = bar_to_molar(self.pref, T)
        self._calc_qelec(T)
        self._calc_qtrans(T)
        self._calc_qrot(T)
        self._calc_qvib(T, ncut=self.ncut)
        self.q['tot'] = self.q['trans'] * self.q['rot'] * self.q['vib']
        self.E['tot'] = self.E['elec'] + self.E['trans'] + self.E['rot'] + \
            self.E['vib']
        # H = E + pV = E + kT per molecule, so that G = H - TS equals the
        # chemical potential at the reference concentration,
        # -kT ln(q/N) (Hermes thesis eqs. 2.17, 2.23). S_trans already
        # contains the +kT from ln N! (Sackur-Tetrode), so without this term
        # G would be kT too low and every equilibrium constant that changes
        # the number of fluid molecules would be off by a factor e each.
        self.H = self.E['tot'] + kB * T
        self.S['tot'] = self.S['elec'] + self.S['trans'] + self.S['rot'] + \
            self.S['vib']

    def get_R(self):
        """Average van der Waals radius in Angstrom (used by DIFF_LIQ)."""
        if self._R is None:
            self._R = calculate_avg_vdw_radius(self.atoms)
        return self._R
    
    R = property(get_R, None)


class Electron(_Thermo):
    """Electrons for electrochemical models.

    Parameters
    ----------
    E : float
        Energy of an electron in eV (e.g. -e times the electrode
        potential).
    self_repulsion : float
        Coefficient in eV of the self-repulsion term,
        lateral = self_repulsion * symbol.
    label : str
        Name of the species.
    """

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
        with conventions(self.conventions):
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
    """Ideal gas: translation (ideal gas), rotation (rigid rotor) and
    vibrations (harmonic oscillators). Its concentration is in M.

    Parameters
    ----------
    atoms : ase.Atoms, ase.db.row.AtomsRow or str
        Structure with a calculator that provides its potential energy
        (eV), or a database row, or the path of a VASP OUTCAR or
        vasprun.xml from a frequency calculation (energy and frequencies
        are then read from it, see micki.io.parse_vasp_out). The geometry
        is used for the moments of inertia.
    label : str
        Name of the species.
    freqs : array of float, optional
        All 3N vibrational frequencies in eV (as from a Hessian), sorted
        ascending; the lowest 6 (5 for linear molecules) are the
        translations and rotations and are dropped. Taken from atoms if it
        is a database row or a VASP file.
    symm : int
        Rotational symmetry number.
    spin : float
        Total spin S; the electronic degeneracy is 2S + 1.
    eref : micki.EnergyReference, optional
        Per-element reference energies subtracted from the potential
        energy.
    rhoref : float, optional
        Reference concentration (M) of the free energy (default 1 M).
    dE : float
        Energy shift in eV (see _Thermo).
    pref : float, optional
        Reference pressure in bar instead of rhoref; the reference
        concentration pref / RT then follows the temperature. pref=1 is
        CatMap's (and ASE's) convention and the default under
        micki.set_conventions('catmap').

    Rates do not depend on the reference state, but free energies, the
    computed alpha of reactions with gases, and barriers clipped with
    Reaction(..., clip=...) do. G = H - TS is the chemical potential at the
    reference state, including the pV = kT term.
    """


class Liquid(_Fluid):
    """Solute or solvent molecule in a liquid, with ideal-gas
    thermochemistry at the reference concentration rhoref (M).

    Parameters are those of Gas (without pref), plus:

    S : float, optional
        Liquid-phase entropy; stored (and saved to databases) but not used
        in the thermochemistry.
    D : float, optional
        Diffusion coefficient in m^2/s, for the DIFF and DIFF_LIQ rate
        laws.
    """

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
        with conventions(self.conventions):
            return self.__class__(self.atoms, label, self.freqs,
                                  self.symm, self.spin, self.eref,
                                  self.rho0, self.Sliq, self.D, self.dE)


class Adsorbate(_Thermo):
    """Adsorbate, transition state or empty site: all degrees of freedom
    are harmonic vibrations (Helmholtz energy).

    Parameters
    ----------
    atoms : ase.Atoms, ase.db.row.AtomsRow or str
        As for Gas. For a bare site (vacancy species) use e.g. the clean
        slab, or an empty Atoms object with energy 0.
    label : str
        Name of the species.
    freqs : array of float
        Vibrational frequencies in eV, all of them used (for a transition
        state, the first, imaginary, mode is dropped). Taken from atoms if
        it is a database row or a VASP file.
    ts : bool
        Whether this is a transition state.
    spin : float
        Total spin S.
    sites : list of Adsorbate
        The site (vacancy) species the adsorbate occupies, once per site,
        e.g. [slab] or [slab, slab] for a bidentate species. Vacancy
        species themselves have no sites. Coverages of each site type add
        up to 1 (or to the site ratio of the lattice).
    lattice : micki.Lattice, optional
        Usually set by the Model (Model.lattice); gives multi-site species
        their configurational entropy kB ln(orientations / symm).
    eref : micki.EnergyReference, optional
        Per-element reference energies subtracted from the potential
        energy.
    dE : float
        Energy shift in eV (see _Thermo).
    symm : int
        Symmetry number: divides the number of distinguishable
        orientations counted by the lattice, e.g. 2 for an end-to-end
        symmetric bidentate species (Hermes et al. 2019, eqs. 6-7). A symm
        larger than the orientations counted (e.g. any symm > 1 for a
        single-site species) only lowers the partition function and warns.
    sitefree : bool
        The species deliberately occupies no sites (e.g. a mobile
        precursor or a dilute-limit model): its coverage is not limited by
        a site balance. Otherwise a Model raises an error for an adsorbate
        without sites that is not itself an empty site, and, with a
        lattice, for a transition state without sites.

    Empty sites are Adsorbates without sites that other species list in
    their sites; nothing else marks them.
    """

    def __init__(self, atoms, label, freqs=None, ts=None,
                 spin=0., sites=None, lattice=None, eref=None, dE=0.,
                 symm=1, sitefree=False):
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
        if sitefree and self.sites:
            raise ValueError('{} has sites and sitefree=True'.format(label))
        self.sitefree = sitefree
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
        # Configurational entropy kB ln(sigma), where sigma is the number of
        # distinguishable orientations: the lattice counts the orientations
        # of multidentate species, and the symmetry number symm divides out
        # those that are identical by symmetry, e.g. symm = 2 for an
        # end-to-end symmetric species (Hermes et al., J. Chem. Phys. 151,
        # 014112 (2019), eqs. 6-7; Hermes thesis eq. 6.1).
        S_conf = 0.
        if self.lattice is not None:
            S_conf = self.lattice.get_S_conf(self.sites)
        if self.symm != 1 and S_conf - kB * np.log(self.symm) < -1e-12 * kB:
            warnings.warn('{}: symm = {} exceeds the number of orientations '
                          'counted by the lattice ({:g}), so its partition '
                          'function is reduced. Orientations are only '
                          'counted for species occupying several sites on a '
                          'lattice.'.format(self.label, self.symm,
                                            np.exp(S_conf / kB)))
        self.S['tot'] += S_conf - kB * np.log(self.symm)


    def copy(self, newlabel=None):
        label = self.label
        if newlabel is not None:
            label = newlabel
        with conventions(self.conventions):
            return self.__class__(self.atoms, label, self.freqs,
                                  self.ts, self.spin, self.sites,
                                  self.lattice, self.eref, self.dE,
                                  self.symm, self.sitefree)


class _Reactants:
    """A sum of species, as built with + and * (e.g. 2 * h + o2); one side
    of a Reaction. Thermodynamic functions are the sums over its species.
    """

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
        # not cached: a gas with a reference pressure has a
        # temperature-dependent reference concentration
        reference_state = 1.
        for species in self.species:
            reference_state *= species.get_reference_state()
        return reference_state

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
