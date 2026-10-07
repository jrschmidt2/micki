import numpy as np
from ase.data import vdw_radii

def calculate_avg_vdw_radius(atoms, npoints=8001):
    if npoints % 2 == 0:
        npoints += 1

    vecs = np.zeros((npoints, 3))

    # Create a Fibonacci spiral
    offset = 2 / npoints
    increment = np.pi * (3 - np.sqrt(5))
    for i in range(npoints):
        y = i * offset - 1 + offset/2
        r = np.sqrt(1 - y**2)
        phi = ((i + 1) % npoints) * increment
        x = np.cos(phi) * r
        z = np.sin(phi) * r
        vecs[i] = np.array([x, y, z])

    natoms = len(atoms)
    pos = atoms.get_positions()
    pos -= atoms.get_center_of_mass()
    rad = np.array([vdw_radii[atom.number] for atom in atoms])

    Rmol = np.zeros(npoints)
    for i, vec in enumerate(vecs):
        # distance from the center of mass to where the ray along vec
        # leaves the outermost vdW sphere it passes through
        for j in range(natoms):
            Rparr = np.dot(pos[j], vec)
            Rperp2 = np.dot(pos[j], pos[j]) - Rparr**2
            Ratom = rad[j]
            # Skip atom if the line doesn't intersect its vdW sphere
            if Rperp2 > Ratom**2:
                continue
            Rmol[i] = max(Rmol[i], Rparr + np.sqrt(Ratom**2 - Rperp2))
    return np.average(Rmol)
