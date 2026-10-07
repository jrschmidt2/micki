"""Module for doing sensitivity analysis of microkinetic model"""

import collections.abc

import numpy as np
import sympy as sym

from ase.units import kB

from micki.reactants import Adsorbate, _Fluid


class ModelAnalysis:
    def __init__(self, model, product_reaction, Uequil, tol=1e-3, dt=3600):
        self.model = model
        self.reaction_name = product_reaction
        self.product_reaction = model.reactions[product_reaction]
        self.Uequil = Uequil
        self.tol = tol
        self.dt = dt

        self.model.set_initial_conditions(self.Uequil)

        t, self.U, self.r = self.model.find_steady_state()
        model.finalize()

        self.species_symbols = []
        for species in self.model._species:
            if species.symbol is not None:
                self.species_symbols.append(species)
        self.rmid = self.r[self.reaction_name]

    def campbell_rate_control(self, rxn_name, scale=0.001):
        reaction = self.model.reactions[rxn_name]

        subs = {}
        for species in self.species_symbols:
            subs[species.symbol] = self.U[species.label]

        kmid = reaction.get_kfor(self.model.T,
                                 self.model.Asite,
                                 self.model.z)

        if isinstance(kmid, sym.Basic):
            kmid = kmid.subs(subs)

        reaction.set_scale('kfor', 1.0 - scale)
        reaction.set_scale('krev', 1.0 - scale)
        reaction.update(self.model.T,
                        self.model.Asite,
                        self.model.z,
                        force=True)
        klow = reaction.get_kfor()
        model = self.model.copy()

        try:
            t1, U1, r1 = model.find_steady_state()
        finally:
            reaction.set_scale('kfor', 1.0)
            reaction.set_scale('krev', 1.0)

        model.finalize()
        rlow = r1[self.reaction_name]
        if isinstance(klow, sym.Basic):
            subs = {}
            for species in self.species_symbols:
                subs[species.symbol] = U1[species.label]
            klow = klow.subs(subs)

        reaction.set_scale('kfor', 1.0 + scale)
        reaction.set_scale('krev', 1.0 + scale)
        reaction.update(self.model.T,
                        self.model.Asite,
                        self.model.z,
                        force=True)
        khigh = reaction.get_kfor()
        model = self.model.copy()

        try:
            t2, U2, r2 = model.find_steady_state()
        finally:
            reaction.set_scale('kfor', 1.0)
            reaction.set_scale('krev', 1.0)

        model.finalize()
        rhigh = r2[self.reaction_name]
        if isinstance(khigh, sym.Basic):
            subs = {}
            for species in self.species_symbols:
                subs[species.symbol] = U2[species.label]
            khigh = khigh.subs(subs)
        reaction.set_scale('kfor', 1.0)
        reaction.set_scale('krev', 1.0)

        return kmid * (rhigh - rlow) / (self.rmid * (khigh - klow))

    def thermodynamic_rate_control(self, names, dg=None):
        """Degree of thermodynamic rate control, -(kT/r) dr/dG.

        The free energies of all species in `names` are shifted together
        by -dg and +dg (default 0.001 kT) and the derivative is taken by
        central differences.
        """
        T = self.model.T
        if dg is None:
            dg = 0.001 * kB * T

        if not isinstance(names, (list, tuple)):
            species = [self.model.species[names]]
        else:
            species = [self.model.species[name] for name in names]

        for sp in species:
            sp.dE -= dg

        for reaction in self.model._reactions:
            reaction.update(T=self.model.T,
                            Asite=self.model.Asite,
                            L=self.model.z,
                            force=True)

        model = self.model.copy(initialize=False)
        model.set_initial_conditions(self.U)

        try:
            t1, U1, r1 = model.find_steady_state()
        finally:
            for sp in species:
                sp.dE += dg

        model.finalize()
        rlow = r1[self.reaction_name]

        for sp in species:
            sp.dE += dg

        for reaction in self.model._reactions:
            reaction.update(T=self.model.T,
                            Asite=self.model.Asite,
                            L=self.model.z,
                            force=True)

        model = self.model.copy(initialize=False)
        model.set_initial_conditions(self.U)

        try:
            t2, U2, r2 = model.find_steady_state()
        finally:
            for sp in species:
                sp.dE -= dg

        for reaction in self.model._reactions:
            reaction.update(T=self.model.T,
                            Asite=self.model.Asite,
                            L=self.model.z,
                            force=True)

        model.finalize()
        rhigh = r2[self.reaction_name]

        # central difference: the two rates are 2 * dg apart in free energy
        return (rlow - rhigh) * kB * T / (self.rmid * 2 * dg)

    def activation_barrier(self, dT=0.01):
        T = self.model.T

        model = self.model.copy(initialize=False)
        model.T = T - dT
        model.set_initial_conditions(self.Uequil)
        t1, U1, r1 = model.find_steady_state()
        model.finalize()

        rlow = r1[self.reaction_name]

        model = self.model.copy(initialize=False)
        model.T = T + dT
        model.set_initial_conditions(self.Uequil)
        t2, U2, r2 = model.find_steady_state()
        model.finalize()

        rhigh = r2[self.reaction_name]

        return kB * T**2 * (rhigh - rlow) / (self.rmid * 2 * dT)

    def rate_order(self, name, drho=0.05):
        species = self.model.species[name]

        rhomid = self.Uequil[species.label]
        if not rhomid > 0:
            raise ValueError('{} has no positive concentration'.format(name))

        U0 = self.Uequil.copy()
        rholow = rhomid * (1.0 - drho)
        U0[species.label] = rholow
        model = self.model.copy(initialize=False)
        model.set_initial_conditions(U0)
        t1, U1, r1 = model.find_steady_state()
        model.finalize()
        rlow = r1[self.reaction_name]

        rhohigh = rhomid * (1.0 + drho)
        U0[species.label] = rhohigh
        model.set_initial_conditions(U0)
        t2, U2, r2 = model.find_steady_state()
        model.finalize()
        rhigh = r2[self.reaction_name]

        return (rhomid / self.rmid) * (rhigh - rlow) / (rhohigh - rholow)

    def drate_order_dg(self, fluid, adsorbates, rho_scale=0.01, g_scale=0.01):
        """(rho/r) d^2 r / (d rho dG): the change of the rate order in
        `fluid` with the free energy of `adsorbates` (shifted together).

        Central differences in the concentration of `fluid` (+-drho) and in
        the free energies of `adsorbates` (+-dg).
        """
        if not isinstance(fluid, _Fluid):
            raise TypeError('fluid must be a Gas or Liquid species')

        if not isinstance(adsorbates, collections.abc.Iterable):
            adsorbates = [adsorbates]

        if not isinstance(adsorbates[0], Adsorbate):
            raise TypeError('adsorbates must be Adsorbate species')

        rhomid = self.U[fluid.label]
        if not rhomid > 0:
            raise ValueError('{} has no positive concentration'
                             ''.format(fluid.label))
        rmid = self.r[self.reaction_name]
        gmid = adsorbates[0].get_G(self.model.T)
        if isinstance(gmid, sym.Basic):
            trans = {}
            for species in self.model._species:
                if isinstance(species, Adsorbate) \
                        and species.symbol is not None:
                    trans[species.symbol] = self.U[species.label]
            gmid = gmid.subs(trans)
        dg = np.abs(gmid * g_scale * 2)
        drho = rhomid * rho_scale * 2

        dr = 0

        def set_dg(species, dg):
            species.dE += dg

        for i in [-1, 1]:
            for adsorbate in adsorbates:
                set_dg(adsorbate, i * dg)

            for reaction in self.model._reactions:
                reaction.update(T=self.model.T,
                                Asite=self.model.Asite,
                                L=self.model.z,
                                force=True)

            for j in [-1, 1]:
                U0 = self.Uequil.copy()
                U0[fluid.label] = rhomid + j * drho

                model = self.model.copy(initialize=False)
                model.set_initial_conditions(U0)

                ti, Ui, ri = model.find_steady_state()
                dr += i * j * ri[self.reaction_name]

            for adsorbate in adsorbates:
                set_dg(adsorbate, -i * dg)

        for reaction in self.model._reactions:
            reaction.update(T=self.model.T,
                            Asite=self.model.Asite,
                            L=self.model.z,
                            force=True)

        # the four rates are 2 * dg and 2 * drho apart
        return (rhomid / rmid) * dr / (4 * dg * drho)

    def check_converged(self, *vals):
        for val in vals:
            for i, key in enumerate(val[0]):
                if np.abs(val[-1][key] - val[-2][key]) > self.tol:
                    raise ValueError("Calculation not converged ({}: {}, "
                                     "change {})! Increase dt or use better "
                                     "initial guess.".format(
                                         key, val[-1][key],
                                         val[-1][key] - val[-2][key]))
