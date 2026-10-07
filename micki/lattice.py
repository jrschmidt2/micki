"""Lattice stuff"""

from __future__ import division

from .reactants import _Thermo
import numpy as np
from ase.units import kB


class Lattice(object):
    def __init__(self, neighborlist):
        self.neighborlist = neighborlist
        self.sites = [site for site in neighborlist]
        if isinstance(self.sites[0], str):
            self.string_names = True
        elif isinstance(self.sites[0], _Thermo):
            self.string_names = False
        else:
            raise ValueError('All sites must be _Thermo objects or strings!')

        sitetype = str if self.string_names else _Thermo

        for site in self.sites:
            if not isinstance(site, sitetype):
                raise ValueError('All sites must be _Thermo objects or strings!')

        # Sanity check the input
        self.totneighbors = {}
        for site, neighbors in neighborlist.items():
            self.totneighbors[site] = 0
            for neighbor, val in neighbors.items():
                self.totneighbors[site] += val
                if neighbor not in self.sites:
                    raise ValueError("Neighbor {} is unknown!".format(neighbor))
            for site in self.sites:
                if site not in neighbors:
                    neighbors[site] = 0

        # Create a neighbor list matrix
        nsites = len(self.sites)
        if nsites == 1:
            self.ratio = {self.sites[0]: 1}
            return
        nmat = np.zeros((nsites, nsites), dtype=float)
        for i, a in enumerate(self.sites):
            for j, b in enumerate(self.sites):
                nmat[i, j] = self.neighborlist[b].get(a, 0) / self.totneighbors[a]
        
        # Diagonalize to find the element ratio. Only one eigenvector should
        # have all positive values. This is the eigenvector that describes
        # the element ratio
        eigenvals, eigenvecs = np.linalg.eig(nmat)
        for i in range(nsites):
            if np.all(eigenvecs[:, i] < 0) or np.all(eigenvecs[:, i] > 0):
                ratio = np.abs(eigenvecs[:, i])
                ratio /= ratio.sum()
                self.ratio = {site: ratio[i] for i, site in enumerate(self.sites)}
                break
        else:
            print("Eigenvectors: {}".format(eigenvecs))
            raise ValueError("Failed to find the element ratio! Please "
                             "double-check your neighbor count.")

    def update_site_names(self, string_to_thermo):
        if not self.string_names:
            raise RuntimeError('Sites are already _Thermo objects!')

        for site in self.sites:
            if site not in string_to_thermo:
                raise ValueError('No _Thermo object for site {}!'.format(site))

        for string in string_to_thermo:
            if string not in self.sites:
                raise ValueError('Unknown site name {}!'.format(string))

        new_neighborlist = {}
        for site, neighbors in self.neighborlist.items():
            new_neighborlist[string_to_thermo[site]] = {
                string_to_thermo[neighbor]: count
                for neighbor, count in neighbors.items()}

        # rebuild everything (totneighbors, ratio, ...) with the new keys
        self.__init__(new_neighborlist)

    def get_S_conf(self, sites):
        if sites is None or isinstance(sites, _Thermo) or len(sites) == 1:
            return 0
        nconfs = 1
        for i in range(1, len(sites)):
            ncount = self.neighborlist[sites[i - 1]][sites[i]]
            if ncount == 0:
                raise ValueError("This binding geometry is impossible!")
            nconfs *= ncount / self.ratio[sites[i]]
        return kB * np.log(nconfs)
