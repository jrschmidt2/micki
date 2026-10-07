"""Microkinetic modeling objects"""

import warnings

from collections import OrderedDict

import numpy as np
import sympy as sym

from ase.units import kB, _hplanck, kg, _k, _Nav, mol

from micki.reactants import _Thermo, _Fluid, _Reactants, Gas, Liquid, Adsorbate
from micki.reactants import Electron

from micki.lattice import Lattice
from micki.solver import IDASolver


def _at_zero_coverage(expr):
    """Value of a (possibly coverage-dependent) expression at zero
    coverage, as a float."""
    if isinstance(expr, sym.Basic):
        expr = expr.subs({symbol: 0 for symbol in expr.free_symbols})
    return float(expr)


class Reaction:
    """Elementary step reactants <-> products.

    ts: transition state species, rate law method='TST' by default (else
    'EQUIL'). Its free energy follows the coverage-dependent energy
    corrections (lateral interactions and dE) of the reactants and products
    as G_TS + (1 - alpha) sum_reactants + alpha sum_products, where alpha is
    computed self-consistently from the forward and reverse barriers (Hermes
    thesis eqs. 3.58-3.60) unless given. With explicit_ts=True only the TS's
    own energy (including ts.lateral) is used.

    clip: how negative barriers are handled. None: an error for a negative
    barrier at zero coverage (before dE shifts and lateral interactions).
    'zero_coverage' (TST only): at zero coverage, a negative forward
    barrier is set to 0 and a negative reverse barrier makes the forward
    barrier dG; the choice is then kept for all coverages (a forward
    barrier set to 0 loses its coverage dependence). 'coverage' (TST,
    EQUIL, STICK, or a given dG_act): the forward barrier is
    Max(dG_act, dG, 0) at the current coverages, i.e. the transition state
    is raised to the higher of the initial and final states, as in CatMap;
    EQUIL then has the barrier Max(dG, 0) and STICK the collision rate
    times exp(-Max(dG, 0)/kT). dG refers to the reference states of the
    species (see Gas(..., pref=...)). krev = kfor / keq in all cases.
    """

    def __init__(self, reactants, products, ts=None, method=None, S0=1.,
                 dG_act=None, dground=False, reversible=True, clip=None,
                 alpha=None, explicit_ts=False):
        if dground:
            raise ValueError("dground has been replaced by "
                             "clip='zero_coverage'")
        if clip not in (None, 'zero_coverage', 'coverage'):
            raise ValueError("clip must be None, 'zero_coverage' or "
                             "'coverage', not {!r}".format(clip))
        if alpha is not None and not 0. <= alpha <= 1.:
            raise ValueError('alpha must be between 0 and 1')
        if alpha is not None and explicit_ts:
            raise ValueError('Give either alpha or explicit_ts, not both')
        if ts is None and (alpha is not None or explicit_ts
                           or clip == 'zero_coverage'):
            raise ValueError("alpha, explicit_ts and clip='zero_coverage' "
                             "require a transition state")

        # Wrap reactants and products in _Reactants type. _Reactants passed
        # in are copied, since bare sites may be added to them below.
        if isinstance(reactants, _Thermo):
            self.reactants = _Reactants([reactants])
        elif isinstance(reactants, _Reactants):
            self.reactants = reactants.copy()
        else:
            raise NotImplementedError

        if isinstance(products, _Thermo):
            self.products = _Reactants([products])
        elif isinstance(products, _Reactants):
            self.products = products.copy()
        else:
            raise NotImplementedError

        # Determine the number of sites on the LHS and the RHS of the reaction,
        # then add "bare" sites as necessary to balance the site number.
        vacancies = OrderedDict() 
        self.species = []
        for species in self.reactants:
            if species not in self.species:
                self.species.append(species)
            if species.sites is None:
                continue
            for site in species.sites:
                if site in vacancies:
                    vacancies[site] -= 1
                else:
                    vacancies[site] = -1
        for species in self.products:
            if species not in self.species:
                self.species.append(species)
            if species.sites is None:
                continue
            for site in species.sites:
                if site in vacancies:
                    vacancies[site] += 1
                else:
                    vacancies[site] = 1

        # If the user supplied "bare" sites, count them too
        for species in self.reactants:
            if species in vacancies:
                vacancies[species] -= 1
        for species in self.products:
            if species in vacancies:
                vacancies[species] += 1

        for vacancy, nvac in vacancies.items():
            # There are extra sites on the RHS, so add some to the LHS
            if nvac > 0:
                self.reactants += nvac * vacancy
            # There are extra sites on the LHS, so add some to the RHS
            elif nvac < 0:
                self.products += abs(nvac) * vacancy

        self.ts = None
        # The user supplied a transition state species
        if ts is not None:
            if dG_act is not None:
                raise ValueError("Cannot specify both barrier height and "
                                 "transition state!")
            # Wrap the TS in the _Reactants class
            if isinstance(ts, _Thermo):
                self.ts = _Reactants([ts])
            elif isinstance(ts, _Reactants):
                self.ts = ts
            # Fail if the user supplies something other than a _Thermo
            # or _Reactants
            else:
                raise NotImplementedError
        # FIXME: Add stoichiometry checking to ensure logical reactions.
        # Caveat: Don't fail on unbalanced adsorption sites, since some
        # species take up more than one site.

        self.involves_catalyst = False
        for species in self.reactants:
            if isinstance(species, Adsorbate):
                self.involves_catalyst = True
                break
        if not self.involves_catalyst:
            for species in self.products:
                if isinstance(species, Adsorbate):
                    self.involves_catalyst = True
                    break

        self.method = method
        if self.method is None:
            if self.ts is not None:
                self.method = 'TST'
            else:
                self.method = 'EQUIL'

        if isinstance(self.method, str):
            self.method = self.method.upper()

        self.S0 = S0
        self.keq = None
        self.kfor = None
        self.krev = None
        self.T = None
        self.Asite = None
        self.L = None
        self.scale_params = ['dH_act', 'dS_act', 'kfor', 'krev']
        self.alpha = None
        self.reversible = reversible

        # Scaling for sensitivity analysis, defaults to 1 (no scaling)
        self.scale = OrderedDict() 
        for param in self.scale_params:
            self.scale[param] = 1.0
        self.scale_old = self.scale.copy()

        # If the user supplied a TS, this should be None.
        self.dG_act = dG_act

        # If all reactants are Liquid species, then this reaction can occur
        # at any point of the diffusion grid, not just near the catalyst
        # surface.
        self.all_liquid = True

        # Count up the number of Fluid and Adsorbate species on either side
        # of the reaction. This is necessary to construct a proper Jacobian
        # for the rate of change of Fluid species vs Adsorbate species.
        # The Jacobian is related to the concentration in M of catalytic sites
        # in the model (defaults to 1 M in the Model class).
        self.Nreact_fluid = 0
        self.Nreact_ads = 0
        for species in self.reactants:
            if not isinstance(species, Liquid):
                self.all_liquid = False
            if isinstance(species, _Fluid):
                self.Nreact_fluid += 1
            elif isinstance(species, Adsorbate):
                self.Nreact_ads += 1

        self.Nprod_fluid = 0
        self.Nprod_ads = 0
        for species in self.products:
            if not isinstance(species, Liquid):
                self.all_liquid = False
            if isinstance(species, _Fluid):
                self.Nprod_fluid += 1
            elif isinstance(species, Adsorbate):
                self.Nprod_ads += 1

        self.Nfluid = self.Nreact_fluid + self.Nprod_fluid
        self.Nads = self.Nreact_ads + self.Nprod_ads

        self.clip = clip
        self.alpha_fixed = alpha
        self.explicit_ts = explicit_ts
        if clip == 'coverage' and self.method not in ('TST', 'EQUIL',
                                                      'STICK'):
            raise ValueError("clip='coverage' is only implemented for the "
                             "TST, EQUIL and STICK rate laws")

    def _check_scale_param(self, param):
        if param not in self.scale:
            raise ValueError('{} is not a valid scaling parameter; valid '
                             'names are {}'.format(param, self.scale_params))

    def get_scale(self, param):
        self._check_scale_param(param)
        return self.scale[param]

    def set_scale(self, param, value):
        self._check_scale_param(param)
        self.scale[param] = value

    def update(self, T=None, Asite=None, L=None, force=False):
        if not force and not self.is_update_needed(T, Asite, L):
            return

        for species in self.species:
            species.update(T=T, force=True)

        self.T = T
        self.Asite = Asite
        self.L = L
        self.dH = self.products.get_H(T) - self.reactants.get_H(T)
        self.dS = self.products.get_S(T) - self.reactants.get_S(T)
        self.dG = self.dH - self.T * self.dS
        if self.ts is not None:
            for species in self.ts:
                species.update(T=T, force=True)

            Gts = self.ts.get_G(T)
            Gr = self.reactants.get_G(T)
            Gp = self.products.get_G(T)

            dEr = np.sum([species.lateral + species.dE for species in self.reactants])
            dEp = np.sum([species.lateral + species.dE for species in self.products])

            # barriers without dE shifts and coverage dependence
            ts_lateral = sum(species.lateral for species in self.ts)
            dGf = _at_zero_coverage(Gts - Gr + dEr - ts_lateral)
            dGr = _at_zero_coverage(Gts - Gp + dEp - ts_lateral)

            computed_alpha = self.alpha_fixed is None and not self.explicit_ts
            if computed_alpha or self.clip != 'coverage':
                if dGf < 0:
                    raise RuntimeError('Reaction {} has negative forwards activation barrier!'.format(self))
                if dGr < 0:
                    raise RuntimeError('Reaction {} has negative reverse activation barrier!'.format(self))

            if self.explicit_ts:
                self.alpha = None
                shift = 0.
            elif self.alpha_fixed is not None:
                self.alpha = self.alpha_fixed
                shift = (1 - self.alpha) * dEr + self.alpha * dEp
            else:
                all_symbols = set()
                all_symbols.update(sym.sympify(dEr).atoms(sym.Symbol))
                all_symbols.update(sym.sympify(dEp).atoms(sym.Symbol))

                # is_zero, not == 0: sympy >= 1.13 has Float(0.0) != 0
                if sym.sympify(dEp - dEr).subs({symbol: 0 for symbol in all_symbols}).is_zero:
                    self.alpha = dGf / (dGf + dGr)
                else:
                    a1 = (2*dEp - 2*dEr - dGf - dGr - sym.sqrt(8*(dEp-dEr)*dGf + (-2*dEp + 2*dEr + dGf + dGr)**2))/(4*(dEp-dEr))
                    a1 = sym.sympify(a1).subs({symbol: 0 for symbol in all_symbols})
                    if isinstance(a1, sym.Float) and 0. <= a1 <= 1:
                        self.alpha = a1
                    else:
                        a2 = (2*dEp - 2*dEr - dGf - dGr + sym.sqrt(8*(dEp-dEr)*dGf + (-2*dEp + 2*dEr + dGf + dGr)**2))/(4*(dEp-dEr))
                        self.alpha = sym.sympify(a2).subs({symbol: 0 for symbol in all_symbols})
                        if not isinstance(self.alpha, sym.Float) or not (0. <= self.alpha <= 1.):
                            raise RuntimeError("Couldn't find alpha parameter for {}!".format(self))
                shift = (1 - self.alpha) * dEr + self.alpha * dEp

            self.dH_act = self.ts.get_H(T) + shift - self.reactants.get_H(T)
            self.dH_act *= self.scale['dH_act']
            self.dS_act = self.ts.get_S(T) - self.reactants.get_S(T)
            self.dS_act *= self.scale['dS_act']
            self.dG_act = self.dH_act - self.T * self.dS_act

            # If there is a coverage dependence, assume everything has
            # coverage 0
            if self.clip == 'zero_coverage':
                dG_act = _at_zero_coverage(self.dG_act)
                if dG_act < 0.:
                    warnings.warn('Negative activation energy found for {}. '
                                  'Rounding to 0.'.format(self),
                                  RuntimeWarning, stacklevel=2)
                    self.dG_act = 0.

                dG_rev = _at_zero_coverage(self.dG_act - self.dG)
                if dG_rev < 0.:
                    warnings.warn('Negative activation energy found for {}. '
                                  'Rounding to {}'.format(self, self.dG),
                                  RuntimeWarning, stacklevel=2)
                    self.dG_act = self.dG
        self._calc_keq()
        self._calc_kfor()
        self._calc_krev()
        self.scale_old = self.scale.copy()

    def is_update_needed(self, T, Asite, L):
        for species in self.species:
            if species.is_update_needed(T):
                return True
        if self.keq is None:
            return True
        if T is not None and T != self.T:
            return True
        if Asite is not None and Asite != self.Asite:
            return True
        if L is not None and L != self.L:
            return True
        for param in self.scale_params:
            if self.scale[param] != self.scale_old[param]:
                return True
        return False

    def get_keq(self, T=None, Asite=None, L=None):
        self.update(T, Asite, L)
        return self.keq

    def get_kfor(self, T=None, Asite=None, L=None):
        self.update(T, Asite, L)
        return self.kfor

    def get_krev(self, T=None, Asite=None, L=None):
        self.update(T, Asite, L)
        return self.krev

    def _calc_keq(self):
        self.keq = sym.exp(-self.dG / (kB * self.T)) \
                * self.products.get_reference_state() \
                / self.reactants.get_reference_state() \
                * self.scale['kfor'] / self.scale['krev']

    def _calc_kfor(self):
        barr = 1
        if self.clip == 'coverage':
            # CatMap's convention: the transition state is raised to the
            # higher of the initial and final states at the current coverages
            dG_act = 0. if self.dG_act is None else self.dG_act
            barr = sym.exp(-sym.Max(dG_act, self.dG, 0) / (kB * self.T)) \
                / self.reactants.get_reference_state()
        elif self.dG_act is not None:
            barr *= sym.exp(-self.dG_act / (kB * self.T)) \
                    / self.reactants.get_reference_state()
        if self.method == 'EQUIL' and self.clip == 'coverage':
            self.kfor = _k * self.T * barr / _hplanck * self.scale['kfor']
        elif self.method == 'EQUIL':
            self.kfor = _k * self.T * barr / _hplanck * self.scale['kfor']
            if isinstance(self.keq, sym.Basic):
                subs = {}
                for atom in self.keq.atoms(sym.Symbol):
                    subs[atom] = 0.
                keq = self.keq.subs(subs)
            else:
                keq = self.keq
            if keq < 1:
                self.kfor *= self.keq * self.scale['krev'] / self.scale['kfor']
        elif self.method == 'DIEQUIL':
            kfor1 = _k * self.T * barr / _hplanck * self.scale['kfor']
            kfor2 = kfor1 * self.keq * self.scale['krev'] / self.scale['kfor']
            self.kfor = kfor1 * kfor2 / (kfor1 + kfor2)
        elif self.method == 'STICK':
            # STICK is TST with the transition state being a non-interacting
            # 2D ideal gas.
            found_fluid = False
            for species in self.reactants:
                if isinstance(species, _Fluid):
                    if found_fluid:
                        raise ValueError("At most one fluid "
                                         "can react with STICK!")
                    found_fluid = True
                    fluid = species
            if not found_fluid:
                raise ValueError("STICK requires a fluid reactant!")
            Sfluid = fluid.get_S(self.T)
            fluid._calc_qtrans2D(self.T, self.Asite)
            Strans = Sfluid - fluid.S['elec'] - fluid.S['rot'] - fluid.S['vib']
            Slost = Strans / fluid.S['trans']
            dS = (fluid.S['trans2D'] - fluid.S['trans']) * Slost
            dG = fluid.E['trans2D'] - fluid.E['trans'] - self.T * dS
            self.kfor = barr * _k * self.T / _hplanck * np.exp(-dG / (kB * self.T))
            if self.dG_act is None and self.clip != 'coverage':
                # dG above depends on the fluid's reference state through
                # its translational entropy (barr divides by it otherwise)
                self.kfor /= self.reactants.get_reference_state()
            self.kfor *= self.scale['kfor']
        elif self.method == 'ER':
            # Collision Theory
            # kfor = S0 * Asite / (sqrt(2 * pi * m * kB * T))
            m_react = 0.
            for species in self.reactants:
                if isinstance(species, _Fluid):
                    m_react += species.atoms.get_masses().sum()
            kfor1 = barr * 1000 * self.S0 * _Nav * self.Asite \
                * np.sqrt(_k * self.T * kg / (2 * np.pi * m_react)) \
                * self.scale['kfor']

            m_prod = 0.
            for species in self.products:
                if isinstance(species, _Fluid):
                    m_prod += species.atoms.get_masses().sum()
            # FIXME: barr should be different for reverse reaction
            krev2 = barr * 1000 * self.S0 * _Nav * self.Asite \
                * np.sqrt(_k * self.T * kg / (2 * np.pi * m_prod)) \
                * self.scale['krev']
            kfor2 = self.keq * krev2
            self.kfor = kfor1 * kfor2 / (kfor1 + kfor2)
        elif self.method == 'DIFF':
            if self.L is None:
                raise ValueError("Must provide diffusion length "
                                 "for diffusion reactions!")
            found_fluid = False
            for species in self.reactants:
                if isinstance(species, _Fluid):
                    if found_fluid:
                        raise ValueError("Diffusion reaction must "
                                         "have exactly 1 fluid!")
                    found_fluid = True
                    D = species.D
            if not found_fluid or D is None:
                raise ValueError("Diffusion reaction requires a fluid "
                                 "reactant with a diffusion coefficient D!")
            sites = 1
            for species in self.reactants:
                if isinstance(species, Adsorbate):
                    sites *= species.symbol
            for species in self.products:
                if not isinstance(species, (Adsorbate, Electron)):
                    raise ValueError("All products must be adsorbates "
                                     "in diffusion reaction!")
            self.kfor = 1000 * D * self.Asite * mol * barr \
                * self.scale['kfor'] / (self.L * sites)
        elif self.method == 'DIFF_LIQ':
            if len(self.reactants) != 2:
                raise ValueError("DIFF_LIQ rate only defined for reactions "
                                 "with exactly two reactants!")
            if not self.all_liquid:
                raise ValueError("DIFF_LIQ rate only defined for all-liquid "
                                 "phase reactions!")
            Rtot = self.reactants[0].R + self.reactants[1].R
            Dtot = self.reactants[0].D + self.reactants[1].D
            self.kfor = 4 * np.pi * Dtot * Rtot * 1e-10 * 1000 * _Nav
            self.kfor *= self.scale['kfor']
        elif self.method == 'TST':
            # Transition State Theory
            self.kfor = (_k * self.T / _hplanck) * barr * self.scale['kfor']
        else:
            raise ValueError("Method {} is not recognized!".format(
                self.method))

    def _calc_krev(self):
        self.krev = self.kfor / self.keq

    def __repr__(self):
        string = self.reactants.__repr__() + ' <-> '
        if self.ts is not None:
            string += self.ts.__repr__() + ' <-> '
        string += self.products.__repr__()
        return string


class Model:
    def __init__(self, T, Asite, z=0, lattice=None, reactor='CSTR', rhocat=1,
                 analytic_jac=False):
        self.reactions = OrderedDict()
        self._reactions = []
        self._species = []
        self.species = OrderedDict()
        self.vacancy = []
        self.vacspecies = OrderedDict()
        self.solvent = None
        self.fixed = []
        self.initialized = False
        self.U0 = None
        self.rhocat = rhocat
        # Use the symbolically derived Jacobian in the DAE solver instead
        # of IDA's difference-quotient approximation
        self.analytic_jac = analytic_jac

        self.T = T  # System temperature
        self.Asite = Asite  # Area of adsorption site
        self._z = z  # Diffusion length
        self.lattice = lattice
        self.reactor = reactor

    def add_reactions(self, reactions):
        # Set up list of reactions and species
        for name, reaction in reactions.items():
            if not isinstance(reaction, Reaction):
                raise TypeError('{} is not a Reaction'.format(name))
            if reaction in self._reactions:
                continue
            self._reactions.append(reaction)
            self.reactions[name] = reaction
            for species in reaction.species:
                self._add_species(species)
            if reaction.ts is not None:
                for ts in reaction.ts:
                    ts.lattice = self.lattice
            reaction.update(T=self.T, Asite=self.Asite, L=self.z)

    def set_solvent(self, solvent):
        # Solvent will not diffuse even in diffusion system
        if solvent is not None:
            if self.solvent is not None:
                warnings.warn('Overriding old solvent {} with {}.'
                              ''.format(self.solvent, solvent),
                              RuntimeWarning, stacklevel=2)
            if not isinstance(self.species[solvent], Liquid):
                raise ValueError("Solvent must be a Liquid!")
            self.solvent = solvent

    def set_fixed(self, fixed):
        # Fixed species are removed from the differential equations
        if isinstance(fixed, str):
            fixed = [fixed]
        for name in fixed:
            if name not in self.fixed:
                self.fixed.append(name)

    def _add_species(self, species):
        if not isinstance(species, _Thermo):
            raise TypeError('{} is not a species'.format(species))
        # Do nothing if we already know about the species
        if species in self._species or species in self.vacancy:
            return
        # Add the species to the list of known species
        species.lattice = self.lattice
        self.species[species.label] = species
        self._species.append(species)
        # Add the sites that species occupies to the list of known vacancies.
        if species.sites is not None:
            for site in species.sites:
                if site not in self.vacancy:
                    if site in self._species:
                        self._species.remove(site)
                    self.vacancy.append(site)
                    self.vacspecies[site] = [species]
                else:
                    self.vacspecies[site].append(species)

    def set_T(self, T):
        self._T = T
        for reaction in self._reactions:
            reaction.update(T=T, Asite=self.Asite, L=self.z)
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_T(self):
        return self._T

    T = property(get_T, set_T, doc='Model temperature')

    def set_Asite(self, Asite):
        self._Asite = Asite
        for reaction in self._reactions:
            reaction.update(T=self.T, Asite=Asite, L=self.z)
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_Asite(self):
        return self._Asite

    Asite = property(get_Asite, set_Asite, doc='Area of an adsorption site')

    def set_z(self, z):
        self._z = z
        for reaction in self._reactions:
            reaction.update(T=self.T, Asite=self.Asite, L=z)
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_z(self):
        return self._z

    z = property(get_z, set_z, doc='Diffusion length')

    def set_lattice(self, lattice):
        if isinstance(lattice, Lattice) or lattice is None:
            self._lattice = lattice
        elif isinstance(lattice, dict):
            self._lattice = Lattice(lattice)
        else:
            raise ValueError('Unable to parse lattice!')
        # Configurational entropies depend on the lattice, so species and
        # reactions are recomputed.
        for species in self._species + self.vacancy:
            species.lattice = self.lattice
            if species.T is not None:
                species.update(force=True)
        for reaction in self._reactions:
            if reaction.ts is not None:
                for ts in reaction.ts:
                    ts.lattice = self.lattice
            reaction.update(T=self.T, Asite=self.Asite, L=self.z, force=True)
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_lattice(self):
        return self._lattice

    lattice = property(get_lattice, set_lattice, doc='Model lattice')

    def set_initial_conditions(self, U0):
        if self.initialized:
            self.finalize()

        # Reorder species such that Liquid -> Gas -> Adsorbate -> Vacancy
        # Steady-state species go to the end.
        newspecies = []
        for species in self._species:
            if isinstance(species, Liquid):
                newspecies.append(species)
        for species in self._species:
            if isinstance(species, (Gas, Electron)):
                newspecies.append(species)
        for species in self._species:
            if isinstance(species, Adsorbate):
                newspecies.append(species)
        self._species = newspecies

        # Also obtain a list of species that will be variables in the
        # differential equations. This excludes fixed species and empty
        # sites.
        self._variable_species = []
        for species in self._species:
            if species.label not in self.fixed + [self.solvent]:
                self._variable_species.append(species)
        self.nvariables = len(self._variable_species)

        # Start with the incomplete user-provided initial conditions
        self.U0 = U0.copy()

        # Initialize counter for vacancies
        occsites = {species: 0 for species in self.vacancy}

        for name in self.U0:
            try:
                species = self.species[name]
            except KeyError:
                for species in self.vacancy:
                    if species.label == name:
                        break
                else:
                    raise ValueError('Species {} is unknown!'.format(name))

            # Ignore all initial conditions for the number of empty sites
            if species in self.vacancy:
                warnings.warn('Initial condition for vacancy concentration '
                              'ignored.', RuntimeWarning, stacklevel=2)
                continue

            # Throw an error if the user provides the concentration for a
            # species we don't know about
            if species not in self._species:
                raise ValueError("Unknown species {}!".format(species))

            # If the species occupies a site, add its concentration to the
            # occupied sites counter
            if species.sites is not None:
                for site in species.sites:
                    occsites[site] += self.U0[name]

        self.dvacdy = np.zeros((len(self.vacancy), self.nvariables), dtype=int)
        self.vactot = {}
        # Determine what the initial vacancy concentration should be
        for i, vac in enumerate(self.vacancy):
            name = vac.label
            # If a vacancy species is part of the lattice, get its maximum
            # concentration from its relative abundance. Otherwise, assume
            # it is 1.
            if self.lattice is not None and vac in self.lattice.sites:
                self.vactot[vac] = self.lattice.ratio[vac]
            else:
                self.vactot[vac] = 1.
            # Make sure there isn't too much stuff occupying each kind of
            # site on the surface.
            if occsites[vac] > self.vactot[vac]:
                raise ValueError("Too many adsorbates on {}!".format(vac))
            # Normalize the concentration of empty sites to match the
            # appropriate site ratio from the lattice.
            self.U0[name] = self.vactot[vac] - occsites[vac]
            for j, species in enumerate(self._variable_species):
                self.dvacdy[i, j] = -species.sites.count(vac)

        # Populate dictionary of initial conditions for all species
        for name, species in self.species.items():
            # Assume concentration of unnamed species is 0
            if name not in self.U0:
                self.U0[name] = 0.


        # This creates a symbol for each species named modelparamX where X
        # is a three-digit numerical identifier that corresponds to its
        # position in the order of species
        self.symbols_all = []
        # symbols_dict a Thermo object and returns its corresponding symbol
        self.symbols_dict = OrderedDict()
        # symbols ONLY includes species that will be in the differential
        # equations. Fixed species are not included in this list
        self.symbols = []

        for species in self._species:
            self.symbols_all.append(species.symbol)
            self.symbols_dict[species] = species.symbol

        self.symbols = [species.symbol for species in self._variable_species]
        # labels and reaction names in solver order, for converting solver
        # output to dicts
        self._variable_labels = [species.label
                                 for species in self._variable_species]
        rxn_to_name = {rxn: name for name, rxn in self.reactions.items()}
        self._reaction_names = [rxn_to_name[rxn] for rxn in self._reactions]
        
        self.vac_sym = np.zeros(len(self.vacancy), dtype=object)
        # A vacancy will be represented by the total number of sites
        # minus the symbol of each species that occupies one of its sites.
        for i, vacancy in enumerate(self.vacancy):
            self.vac_sym[i] = self.vactot[vacancy]
            for species in self._species:
                self.vac_sym[i] -= species.sites.count(vacancy) * species.symbol

        # known_symbols keeps track of user-provided symbols that the
        # model has seen, so that symbols referring to species not in
        # the model can be later removed.
        known_symbols = set()
        for species in self._species + self.vacancy:
            known_symbols.add(species.symbol)

        # Create the final mass matrix of the proper dimensions
        self.M = np.eye(self.nvariables, dtype=int)
        
        if self.reactor == 'PFR':
            for i, species in enumerate(self._variable_species):
                if isinstance(species, Adsorbate):
                    self.M[i, i] = 0

        # algvar tells the solver which variables are differential
        # and which are algebraic. It is the diagonal of the mass matrix.
        algvar = np.array(self.M.diagonal(), dtype=float)

        # Initialize all rate expressions based on the above symbols
        nrxns = len(self._reactions)
        # Array of symbolic rate expressions
        self.rates = np.zeros(nrxns, dtype=object)
        # Array of rate coefficients.
        self.dypdr = np.zeros((self.nvariables, nrxns), dtype=float)

        for j, rxn in enumerate(self._reactions):
            rate_for = rxn.get_kfor(self.T, self.Asite, self.z)
            rate_rev = rxn.get_krev(self.T, self.Asite, self.z)

            for i, species in enumerate(self._variable_species):
                rcount = rxn.reactants.species.count(species)
                pcount = rxn.products.species.count(species)
                self.dypdr[i, j] = -rcount + pcount
                if isinstance(species, _Fluid) and rxn.involves_catalyst:
                    self.dypdr[i, j] *= self.rhocat

            for species in self._species + self.vacancy:
                rcount = rxn.reactants.species.count(species)
                pcount = rxn.products.species.count(species)
                if not isinstance(species, Electron):
                    rate_for *= species.symbol**rcount
                    rate_rev *= species.symbol**pcount

            # Overall reaction rate (flux)
            self.rates[j] = rate_for
            if rxn.reversible:
                self.rates[j] -= rate_rev


        # Symbols referring to species that are not part of the model can
        # only come from lateral interactions (or other coverage-dependent
        # energies) and are almost certainly a user error.
        known_symbols.update(self.symbols_all)
        rxn_to_name = {rxn: name for name, rxn in self.reactions.items()}
        unknown = OrderedDict()
        for rxn, rate in zip(self._reactions, self.rates):
            for symbol in sorted(sym.sympify(rate).atoms(sym.Symbol), key=str):
                if symbol not in known_symbols:
                    unknown.setdefault(str(symbol), []).append(rxn_to_name[rxn])
        if unknown:
            raise ValueError(
                'Rate expressions depend on species that are not in the '
                'model (check lateral interactions): ' +
                '; '.join('{} (in reaction{} {})'.format(
                    label, 's' if len(names) > 1 else '', ', '.join(names))
                          for label, names in unknown.items()))

        # Fixed species (and the solvent) keep their symbols; their
        # concentrations are passed to the solver at evaluation time.
        fixed_species = [species for species in self._species
                         if species.label in self.fixed
                         or species.label == self.solvent]

        for i, r in enumerate(self.rates):
            self.rates[i] = sym.sympify(r)

        self._solver = IDASolver(self.symbols,
                                 [vac.symbol for vac in self.vacancy],
                                 [sym.sympify(expr) for expr in self.vac_sym],
                                 self.rates, self.dypdr, algvar,
                                 [species.symbol for species in fixed_species],
                                 [self.U0[species.label]
                                  for species in fixed_species])

        # Initial values of the variable species, in solver order
        U0 = []
        for symbol in self.symbols:
            for species, isymbol in self.symbols_dict.items():
                if symbol == isymbol:
                    U0.append(self.U0[species.label])
                    break

        atol = np.array([1e-32] * self.nvariables)
        atol += 1e-16 * algvar
        self._solver.initialize(U0, 1e-10, atol,
                                analytic_jac=self.analytic_jac)

        self.initialized = True

    def _out_array_to_dict(self, U, dU, r):
        Ui = {}
        dUi = {}
        fixed = list(self.fixed)
        if self.solvent is not None:
            fixed.append(self.solvent)
        for name in fixed:
            dUi[name] = 0.
            Ui[name] = self.U0[name]
        for label, Uj, dUj in zip(self._variable_labels, U, dU):
            Ui[label] = Uj
            dUi[label] = dUj
        for vacancy in self.vacancy:
            Ui[vacancy.label] = self.vactot[vacancy]
            for species in self.vacspecies[vacancy]:
                Ui[vacancy.label] -= Ui[species.label]
            dUi[vacancy.label] = 0

        ri = dict(zip(self._reaction_names, r))

        return Ui, dUi, ri

    def find_steady_state(self, dt=60, maxiter=2000, epsilon=1e-8,
                          method='hybrid'):
        """Find the steady state starting from the initial conditions.

        By default the steady-state equations are solved directly with
        Newton's method, falling back to time integration (in steps of dt,
        up to maxiter steps, until max |dy/dt| < epsilon) if Newton does
        not converge from the initial conditions. method='integrate' only
        integrates. self.steady_state_method records the path taken
        ('newton', 'integrate+newton' or 'integrate'); the returned t is
        inf when Newton alone was used.
        """
        t, U1, dU1, r1 = self._solver.find_steady_state(dt, maxiter, epsilon,
                                                        method)
        self.steady_state_method = self._solver.steady_state_method
        self.t = t
        self.U = []
        self.dU = []
        self.r = []
        U, dU, r = self._out_array_to_dict(U1, dU1, r1)
        self.U.append(U)
        self.dU.append(dU)
        self.r.append(r)
        self.check_rates(U)
        return t, U, r

    def solve(self, t, ncp):
        self.t, self.U1, self.dU1, self.r1 = self._solver.solve(ncp, t)
        self.U = []
        self.dU = []
        self.r = []
        for i, t in enumerate(self.t):
            Ui, dUi, ri = self._out_array_to_dict(self.U1[i], self.dU1[i],
                                                  self.r1[i])
            self.U.append(Ui)
            self.dU.append(dUi)
            self.r.append(ri)
        self.check_rates(self.U[-1])
        return self.U, self.r

    def finalize(self):
        self.initialized = False

    def _eval_rate_constant(self, k, symbol_to_coverage):
        # Rate constants may depend on coverages (lateral interactions).
        # Evaluating them with sympy's subs() is slow, so each expression is
        # turned into a NumPy function once and cached.
        if not isinstance(k, sym.Basic):
            return float(k)
        cache = self.__dict__.setdefault('_rate_constant_cache', {})
        entry = cache.get(id(k))
        if entry is None or entry[0] is not k:
            symbols = sorted(k.free_symbols, key=str)
            entry = (k, symbols, sym.lambdify(symbols, k, 'numpy'))
            cache[id(k)] = entry
        _, symbols, func = entry
        try:
            return float(func(*[symbol_to_coverage[x] for x in symbols]))
        except KeyError:
            return sym.sympify(k).subs(symbol_to_coverage)

    def check_rates(self, U, epsilon=1e-6):
        symbol_to_coverage = {}
        for name, Ui in U.items():
            if name in self.species and self.species[name].symbol is not None:
                symbol_to_coverage[self.species[name].symbol] = Ui

        for species in self.vacancy:
            if species.label in U and species.symbol is not None:
                symbol_to_coverage[species.symbol] = U[species.label]

        kmax = _k * self.T / _hplanck
        for name, reaction in self.reactions.items():
            kfor = self._eval_rate_constant(reaction.kfor, symbol_to_coverage)
            krev = self._eval_rate_constant(reaction.krev, symbol_to_coverage)
            for k, word in [(kfor, "Forwards"), (krev, "Reverse")]:
                ratio = k / kmax
                if (ratio - 1.0) > epsilon:
                    warnings.warn(word + " rate constant for {} is too large! "
                                  "Value is {} kB T / h (should be <= 1)."
                                  "".format(reaction, ratio),
                                  RuntimeWarning, stacklevel=2)

    def copy(self, initialize=True):
        newmodel = Model(self.T, self.Asite, z=self.z, lattice=self.lattice,
                         reactor=self.reactor, rhocat=self.rhocat,
                         analytic_jac=self.analytic_jac)
        newmodel.add_reactions(self.reactions)
        newmodel.set_fixed(self.fixed)
        newmodel.set_solvent(self.solvent)
        if initialize:
            newmodel.set_initial_conditions(self.U0)
        return newmodel
