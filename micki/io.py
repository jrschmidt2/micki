"""Reading structures and frequencies from VASP output"""

import numpy as np

from ase.io import read
from ase.units import _hplanck, J, m, kg

from micki.masses import masses

def parse_vasp_out(filename, ignore_atoms=()):
    """Structure and vibrational frequencies from a VASP frequency
    calculation (OUTCAR or vasprun.xml), as (atoms, freqs).

    The Hessian is read and diagonalized with micki's masses, so that all
    species use the same masses and unit conversions. freqs are in eV,
    sorted ascending, imaginary modes as negative values. ignore_atoms
    (indices, Atom objects or element symbols) are left out of the Hessian
    (partial Hessian, e.g. to drop metal atoms).
    """
    atoms = read(filename, index=0)
    for atom in atoms:
        atom.mass = masses[atom.symbol]
    if 'OUTCAR' in filename:
        # This reads the hessian from the OUTCAR and diagonalizes it
        # to find the frequencies, rather than reading the frequencies
        # directly from the OUTCAR. This is to ensure we use the same
        # unit conversion factors, and also to make sure we use the same
        # atom masses for all calculations. Also, allows for the possibility
        # of doing partial hessian diagonalization should we want to do that.
        hessblock = 0
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line != '':
                    if hessblock == 1:
                        if line.startswith('---'):
                            hessblock = 2

                    elif hessblock == 2:
                        line = line.split()
                        dof = len(line)
                        hess = np.zeros((dof, dof), dtype=float)
                        index = np.zeros(dof, dtype=int)
                        cart = np.zeros(dof, dtype=int)
                        for i, direction in enumerate(line):
                            index[i] = int(direction[:-1]) - 1
                            if direction[-1] == 'X':
                                cart[i] = 0
                            elif direction[-1] == 'Y':
                                cart[i] = 1
                            elif direction[-1] == 'Z':
                                cart[i] = 2
                            else:
                                raise ValueError("Error reading Hessian!")
                        hessblock = 3
                        j = 0

                    elif hessblock == 3:
                        line = line.split()
                        hess[j] = np.array([float(val) for val in line[1:]],
                                           dtype=float)
                        j += 1

                    elif line.startswith('SECOND DERIVATIVES'):
                        hessblock = 1

                elif hessblock == 3:
                    break

        hess = -(hess + hess.T) / 2.
    elif 'vasprun.xml' in filename:
        import xml.etree.ElementTree as ET

        tree = ET.parse(filename)
        root = tree.getroot()

        vasp_mass = {}

        for element in root.find("atominfo/array[@name='atomtypes']/set"):
            vasp_mass[element[1].text.strip()] = float(element[2].text)

        selective = np.ones((len(atoms), 3), dtype=bool)
        constblock = root.find(
                'structure[@name="initialpos"]/varray[@name="selective"]')
        if constblock is not None:
            for i, v in enumerate(constblock):
                for j, fixed in enumerate(v.text.split()):
                    selective[i, j] = (fixed == 'T')
        index = []
        for i, atom in enumerate(atoms):
            for direction in selective[i]:
                if direction:
                    index.append(i)

        hess = np.zeros((len(index), len(index)), dtype=float)

        for i, v in enumerate(root.find(
                'calculation/dynmat/varray[@name="hessian"]')):
            hess[i] = -np.array([float(val) for val in v.text.split()])

        vasp_massvec = np.zeros(len(index), dtype=float)
        for i, j in enumerate(index):
            vasp_massvec[i] = vasp_mass[atoms[j].symbol]

        # VASP uses weird masses, so we un-mass-weight here
        hess *= np.sqrt(np.outer(vasp_massvec, vasp_massvec))

    else:
        raise ValueError('Unknown file format {}!'.format(filename))
    mass = np.array([atoms[i].mass for i in index], dtype=float)
    hess /= np.sqrt(np.outer(mass, mass))

    # Temporary work around: My test system OUTCARs include some
    # metal atoms in the hessian, this seems to cause some problems
    # with the MKM. So, here I'm taking only the non-metal part
    # of the hessian and diagonalizing that.
    partial = []
    for i, j in enumerate(index):
        if (j in ignore_atoms
                or atoms[j] in ignore_atoms
                or atoms[j].symbol in ignore_atoms):
            continue
        partial.append(i)
    if not partial:
        return atoms, np.array([])
    partial_hess = np.zeros((len(partial), len(partial)))
    partial_index = np.zeros(len(partial), dtype=int)
    for i, a in enumerate(partial):
        partial_index[i] = index[a]
        for j, b in enumerate(partial):
            partial_hess[i, j] = hess[a, b]

    partial_hess *= _hplanck**2 * J * m**2 * kg / (4 * np.pi**2)
    v, w = np.linalg.eig(partial_hess)

    # We're taking the square root of an array that could include
    # negative numbers, so the result has to be complex.
    freq = np.sqrt(np.array(v, dtype=complex))
    freqs = np.zeros_like(freq, dtype=float)

    # We don't want to deal with complex numbers, so we just convert
    # imaginary numbers to negative reals.
    for i, val in enumerate(freq):
        if val.imag == 0:
            freqs[i] = val.real
        else:
            freqs[i] = -val.imag
    freqs.sort()
    return atoms, freqs
