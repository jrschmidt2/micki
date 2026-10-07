"""Reference atoms object"""

import numpy as np

from ase import Atoms
from ase.io import read
from ase.data import chemical_symbols
from ase.db.row import AtomsRow

from micki.reactants import _Thermo


class EnergyReference(dict):
    """Construct an atomic energy reference.

    This routine accepts an iterable containing N paths to ASE-readable
    geometry files with N unique elements between them, and returns a dict-like
    containing those unique elements as keys and their reference energy as
    values.

    By default, use the same geometry files as those which contain the
    vibrational frequencies for the microkinetic model.
    """
    def __init__(self, species, index=0):
        dict.__init__(self)
        symbols = []
        energies = []
        elements = set()

        self.initialized = False

        for sp in species:
            if isinstance(sp, Atoms):
                conf = sp
            elif isinstance(sp, AtomsRow):
                conf = sp.toatoms()
            elif isinstance(sp, _Thermo):
                conf = sp.atoms
            else:
                conf = read(sp, index=index)
            symbols.append(conf.get_chemical_symbols())
            elements = elements.union(symbols[-1])
            energies.append(conf.get_potential_energy())

        size = len(elements)
        if len(energies) < size:
            raise ValueError("System is underdetermined!")
        elif len(energies) > size:
            raise ValueError("System is overdetermined!")

        coeff = np.zeros((size, size), dtype=float)

        for i in range(size):
            for j, symbol in enumerate(elements):
                coeff[i, j] = symbols[i].count(symbol)
        eref = np.linalg.solve(coeff, energies)

        for i, symbol in enumerate(elements):
            self[symbol] = eref[i]

        self.initialized = True

    def __setitem__(self, key, value):
        if self.initialized:
            raise NotImplementedError
        else:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        raise NotImplementedError

    def __getitem__(self, key):
        if isinstance(key, str):
            key = key.capitalize()
        elif isinstance(key, int):
            key = chemical_symbols[key]
        return super().__getitem__(key)

    def copy(self):
        new = EnergyReference.__new__(EnergyReference)
        new.initialized = False
        for key in self:
            new[key] = self[key]
        new.initialized = True
        return new
