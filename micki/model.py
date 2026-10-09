"""Microkinetic modeling objects"""

import re
import warnings

from collections import Counter, OrderedDict
from collections.abc import Mapping

import numpy as np
import sympy as sym

from ase.units import kB, _hplanck, kg, _k, _Nav, mol

from micki.reactants import _Thermo, _Fluid, _Reactants, Gas, Liquid, Adsorbate
from micki.reactants import Electron

from micki.lattice import Lattice
from micki.conventions import DEFAULT, get_conventions
from micki.solver import IDASolver


def _at_zero_coverage(expr):
    """Value of a (possibly coverage-dependent) expression at zero
    coverage, as a float."""
    if isinstance(expr, sym.Basic):
        expr = expr.subs({symbol: 0 for symbol in expr.free_symbols})
    return float(expr)


def _site_species(species):
    """The empty-site species among `species` and their sites: the species
    that some species lists in its sites (as Model treats them), by
    identity."""
    sites = {}
    for sp in species:
        for site in sp.sites or []:
            sites[id(site)] = site
    return sites


def _composition(species):
    """Element counts of a species without the surface: if an adsorbate's
    structure contains the atoms of its first site species (e.g. the slab),
    they are removed once."""
    if species.atoms is None:
        return Counter()
    counts = Counter(species.atoms.get_chemical_symbols())
    if isinstance(species, Adsorbate) and species.sites \
            and species.sites[0].atoms is not None:
        site = Counter(species.sites[0].atoms.get_chemical_symbols())
        if site and all(counts[e] >= n for e, n in site.items()):
            counts -= site
    return counts


def _side_composition(side, sites):
    """Element counts of a reaction side, without empty sites (`sites`, from
    _site_species) and electrons."""
    total = Counter()
    for species in side:
        if id(species) not in sites and not isinstance(species, Electron):
            total += _composition(species)
    return total


def _formula(counts):
    return ''.join(e + (str(n) if n > 1 else '')
                   for e, n in sorted(counts.items())) or 'nothing'


_ARROW = re.compile(r'<->|->')
_BARRIER = re.compile(r'\^\s*([-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?)\s*(?:eV)?(?:_\w+)?')
_COEFFICIENT = re.compile(r'(\d+)(.*)$', re.S)


def _lookup(label, species):
    """The species called `label`: a key of `species`, or '*' for the only
    empty-site species among them."""
    if label in species:
        return species[label]
    if label == '*':
        sites = list(_site_species(species.values()).values())
        if len(sites) != 1:
            raise ValueError("'*' needs exactly one empty-site species, "
                             'found {}'.format(
                                 ', '.join(sorted(map(str, sites)))
                                 or 'none'))
        return sites[0]
    return None


def _parse_term(term, species, expression):
    """'2 h', '2*h', '2h' -> (2, h); a term that is a label is a label (so
    are CatMap's '*_s' and '2*_s' -> (2, '*_s')); '*' and '2*' are the
    empty site."""
    sp = _lookup(term, species)
    if sp is not None:
        return 1, sp
    match = _COEFFICIENT.match(term)
    if match:
        n, rest = int(match.group(1)), match.group(2).strip()
        for label in (rest, rest[1:].strip() if rest[:1] == '*' else None):
            sp = _lookup(label, species) if label else None
            if sp is not None:
                if n < 1:
                    raise ValueError('Coefficient 0 in {!r}'.format(
                        expression))
                return n, sp
    if not term or any(c.isspace() for c in term) or match:
        raise ValueError('Cannot parse the term {!r} in {!r}'.format(
            term, expression))
    raise ValueError('Unknown species {!r} in {!r}'.format(term, expression))


def _parse_side(text, species, expression):
    """'2 h + o' -> [h, h, o] (species looked up by label)."""
    out = []
    for term in text.split('+'):
        n, sp = _parse_term(term.strip(), species, expression)
        out += [sp] * n
    return out


def _species_of(side):
    """The species of a reaction side (a species or a sum of species)."""
    if isinstance(side, _Thermo):
        return [side]
    if isinstance(side, _Reactants):
        return list(side.species)
    return []


def _sum(species):
    return species[0] if len(species) == 1 else _Reactants(species)


class Reaction:
    """Elementary step reactants <-> products.

    Usually written as a string, Reaction.from_string('co + o <-> o-co ->
    co2_g', species), or several at once with micki.reactions_from_strings.
    The constructor takes the sides as species objects: species or sums of
    species (e.g. 2 * sp['h'] + sp['o2']);
    empty sites needed to balance the sites of both sides are added
    automatically (the vacancy species of the adsorbates' sites).

    Parameters
    ----------
    reactants, products : species or sum of species
    ts : Adsorbate or sum of species, optional
        Transition state. Its free energy follows the coverage-dependent
        energy corrections (lateral interactions + dE) of the reactants and
        products as G_TS + (1 - alpha) sum_reactants + alpha sum_products.
    method : str, optional
        Rate law; default 'TST' with a transition state (or dG_act), else
        'EQUIL'. Rate constants are per site and second, with fluid
        concentrations in M (kfor in 1/(M^n s) for n fluid reactants).

        - 'TST': transition state theory, kfor = kT/h exp(-dG_act/kT)
          (from ts, or from dG_act).
        - 'EQUIL': no barrier: kfor = kT/h, times keq if keq < 1 at zero
          coverage (the barrier is max(0, dG) at zero coverage).
        - 'DIEQUIL': like EQUIL but smooth, 1/kfor = 1/k1 + 1/(k1 keq)
          with k1 = kT/h.
        - 'STICK': adsorption of one fluid by collision theory (TST with
          a 2D-ideal-gas transition state, sticking coefficient 1).
        - 'ER': Eley-Rideal, collision theory with sticking coefficient S0,
          combined with the reverse direction.
        - 'DIFF': diffusion of a fluid through a stagnant layer of
          thickness Model.z to the surface (needs the fluid's D).
        - 'DIFF_LIQ': diffusion-limited reaction of two Liquid species
          (Smoluchowski; needs D and the van der Waals radii).

        krev = kfor / keq for all rate laws (detailed balance).
    S0 : float
        Sticking coefficient for 'ER'.
    dG_act : float, optional
        Activation free energy in eV, instead of a transition state.
    dground : bool
        Removed; raises a ValueError. Use clip='zero_coverage'.
    reversible : bool
        If False, the reverse rate is omitted from the rate expression.
    clip : None, 'zero_coverage' or 'coverage'
        How negative barriers are handled. None (default): an error for a
        negative barrier at zero coverage (computed without dE shifts and
        lateral interactions). 'zero_coverage' (TST only): at zero
        coverage, a negative forward barrier is set to 0 and a negative
        reverse barrier makes the forward barrier dG; the choice is then
        kept for all coverages (a forward barrier set to 0 loses its
        coverage dependence). 'coverage' (TST, EQUIL, STICK, or a given
        dG_act): the forward barrier is Max(dG_act, dG, 0) at the current
        coverages, i.e. the transition state is raised to the higher of the
        initial and final states, as in CatMap; EQUIL then has the barrier
        Max(dG, 0) and STICK the collision rate times exp(-Max(dG, 0)/kT).
        dG refers to the reference states of the species (see Gas(...,
        pref=...)).
    alpha : float or None
        Weight (0 to 1) with which the transition state follows the
        products' energy corrections. None (default): computed
        self-consistently from the forward and reverse barriers
        (BEP-like; Hermes thesis eqs. 3.58-3.60). CatMap's initial_state,
        intermediate_state and final_state are 0, 0.5 and 1.
    explicit_ts : bool
        Use only the transition state's own energy, including ts.lateral
        (no interpolation between reactants and products).
    check_balance : bool
        Raise a ValueError if the reactants, products and transition state
        do not contain the same atoms, or do not carry the same charge.
    beta : float
        Charge-transfer coefficient (symmetry factor) of an electrochemical
        step (default 0.5), for the step as written: the forward barrier
        changes by beta times the change of the reaction free energy with
        the potential, the reverse barrier by 1 - beta. Also the slope of
        dG_act0.
    U_ref : float
        For an electrochemical step with a transition state (or dG_act): the
        electrode potential (V vs SHE) at which its free energy (barrier)
        was computed, e.g. by constant-potential DFT. At potential U,
        dG_act(U) = dG_act(U_ref) + beta (dG(U) - dG(U_ref)).
    dG_act0 : float
        Barrier (eV) where the reaction free energy is zero, instead of a
        transition state: dG_act = dG_act0 + beta dG (e.g. "simple" PCET
        steps, where dG_act0 is the barrier at the step's equilibrium
        potential). In reaction strings: 'a + h3o + e <-> ^0.26 -> ah + h2o'.
    dG_reorg : float
        Additional barrier (eV), e.g. for solvent reorganization, which
        computed barriers miss (rate constants times exp(-dG_reorg/kT));
        added to any rate law, also to barrierless (EQUIL) steps.
    prefactor : float
        Replaces kT/h (1/s) in the TST and EQUIL rate laws (e.g. to mirror
        CatMap models).

    Electrochemical steps (with Electron species, n_electrons != 0) default
    to clip='coverage' and, with a transition state, alpha=beta. Empty sites (species that a species
        of the reaction lists in its sites) and electrons are not counted,
        and an adsorbate whose structure contains the atoms of its (first)
        site species, e.g. a slab, is counted without them.

    Under micki.set_conventions('catmap'), clip defaults to 'coverage' (for
    TST, EQUIL and STICK), alpha to 0.5 (unless explicit_ts), and the
    transition state follows only the lateral interactions of the
    reactants and products, not their dE (see micki.conventions).
    clip=None and alpha=None given explicitly mean no clipping and a
    computed alpha.

    Attributes
    ----------
    keq, kfor, krev : float or sympy expression
        Equilibrium and rate constants at the last update (sympy
        expressions in the coverages with lateral interactions).
    dG, dH, dS : reaction free energy (eV), enthalpy, entropy
    dG_act, dH_act, dS_act : activation free energy, enthalpy, entropy
    alpha : the transition-state weight used
    scale : dict
        Multipliers of 'dH_act', 'dS_act', 'kfor' and 'krev' (set_scale),
        for sensitivity analysis.
    """

    def __init__(self, reactants, products, ts=None, method=None, S0=1.,
                 dG_act=None, dground=False, reversible=True, clip=DEFAULT,
                 alpha=DEFAULT, explicit_ts=False, check_balance=True,
                 beta=None, U_ref=None, dG_act0=None, dG_reorg=0.,
                 prefactor=None, delta=None):
        self.conventions = get_conventions()
        catmap = self.conventions == 'catmap'
        n_electrons = (sum(isinstance(sp, Electron)
                           for sp in _species_of(reactants))
                       - sum(isinstance(sp, Electron)
                             for sp in _species_of(products)))
        electrochemical = n_electrons != 0
        if beta is None and electrochemical:
            beta = 0.5
        if beta is not None and not 0. <= beta <= 1.:
            raise ValueError('beta must be between 0 and 1')
        if dG_act0 is not None:
            if ts is not None or dG_act is not None:
                raise ValueError('Give dG_act0, dG_act or ts, not several')
            if beta is None:
                raise ValueError('dG_act0 (barrier at dG = 0) needs beta')
        if electrochemical and (ts is not None or dG_act is not None) \
                and U_ref is None:
            raise ValueError('An electrochemical step with a transition '
                             'state or dG_act needs U_ref, the potential '
                             '(V vs SHE) at which its barrier was computed')
        if U_ref is not None and not electrochemical:
            raise ValueError('U_ref is for electrochemical steps')
        if clip is DEFAULT:
            clip_default = True
            clip = 'coverage' if catmap or electrochemical else None
        else:
            clip_default = False
        if alpha is DEFAULT:
            if ts is None or explicit_ts:
                alpha = None
            elif catmap:
                alpha = 0.5
            elif electrochemical:
                alpha = beta
            else:
                alpha = None
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
        if check_balance:
            self._check_balance()

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
            if self.ts is not None or dG_act is not None \
                    or dG_act0 is not None:
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
        self.U = 0.  # electrode potential vs SHE (V), set by the Model
        # electrons consumed as written (positive for a reduction)
        self.n_electrons = n_electrons
        self.beta = beta
        self.U_ref = U_ref
        self.dG_act0 = dG_act0
        self.dG_reorg = dG_reorg
        self.prefactor = prefactor
        if prefactor is not None and self.method not in ('TST', 'EQUIL'):
            raise ValueError('prefactor replaces kT/h of the TST and EQUIL '
                             'rate laws')
        # film transport (method 'FILM')
        self.delta = delta
        self.roughness = 1.
        self.model_delta = None
        if self.method == 'FILM':
            if len(self.reactants) != 1 or len(self.products) != 1 \
                    or not isinstance(self.reactants[0], _Fluid) \
                    or not isinstance(self.products[0], _Fluid):
                raise ValueError('FILM transports one fluid species to its '
                                 'near-surface copy: write "bulk -> near"')
        # rates per site: surface steps and film transport (their fluid
        # concentrations change with rhocat, the site concentration)
        self.per_site = self.involves_catalyst or self.method == 'FILM'
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
        self.dG_act_given = dG_act

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

        if clip_default and self.method not in ('TST', 'EQUIL', 'STICK'):
            clip = None
        self.clip = clip
        self.alpha_fixed = alpha
        self.explicit_ts = explicit_ts
        # under CatMap's conventions, a TS does not follow dE shifts
        self.ts_follows_dE = not catmap
        if clip == 'coverage' and self.method not in ('TST', 'EQUIL',
                                                      'STICK'):
            raise ValueError("clip='coverage' is only implemented for the "
                             "TST, EQUIL and STICK rate laws")

    @classmethod
    def from_string(cls, expression, species, **kwargs):
        """A Reaction from a string such as 'co + o <-> o-co -> co2_g'.

        expression: 'reactants -> products' or, with a transition state,
        'reactants <-> ts -> products' (CatMap's form; '->' and '<->' are
        interchangeable and both mean a reversible step, see reversible).
        Instead of a transition state, '^0.26' (or CatMap's '^0.26eV_a')
        gives the barrier dG_act0 at dG = 0.
        Terms are species labels separated by '+', each optionally
        preceded by an integer coefficient ('2 h', '2*h' or '2h'; a term
        that is itself a label is taken as the label). Empty-site species
        (species that others list in their sites) may be written or left
        out, since sites are balanced automatically; '*' stands for the
        empty site if there is only one kind among `species` (an explicit
        '*' key takes precedence). In the transition-state part empty sites
        are ignored.

        species: a dict {label: species} (e.g. from read_from_db) or a list
        of species. Other keyword arguments are passed to Reaction.
        """
        if not isinstance(species, Mapping):
            species = {s.label: s for s in species}
        parts = _ARROW.split(expression)
        if len(parts) == 3 and parts[1].strip().startswith('^'):
            # '^0.26' or CatMap's '^0.26eV_a': barrier at dG = 0 (dG_act0)
            match = _BARRIER.fullmatch(parts[1].strip())
            if match is None:
                raise ValueError('Cannot read the barrier {!r} in {!r}'
                                 ''.format(parts[1].strip(), expression))
            if 'dG_act0' in kwargs:
                raise ValueError('Barrier given twice for {!r}'.format(
                    expression))
            kwargs['dG_act0'] = float(match.group(1))
            parts = [parts[0], parts[2]]
        states = [_parse_side(text, species, expression) for text in parts]
        if len(states) not in (2, 3):
            raise ValueError("Write 'reactants -> products' or 'reactants "
                             "<-> ts -> products', not {!r}".format(
                                 expression))
        if len(states) == 3:
            if 'ts' in kwargs:
                raise ValueError('Transition state given twice for {!r}'
                                 ''.format(expression))
            sites = _site_species(sum(states, []))
            ts = [s for s in states[1] if id(s) not in sites]
            if not ts:
                raise ValueError('No transition state in {!r}'.format(
                    expression))
            kwargs['ts'] = _sum(ts)
        return cls(_sum(states[0]), _sum(states[-1]), **kwargs)

    def _check_balance(self):
        """Raise a ValueError if the reaction does not conserve atoms."""
        sites = _site_species(list(self.reactants) + list(self.products)
                              + list(self.ts or []))
        initial = _side_composition(self.reactants, sites)
        final = _side_composition(self.products, sites)
        states = 'reactants {}, products {}'.format(_formula(initial),
                                                    _formula(final))
        balanced = initial == final
        if self.ts is not None:
            ts = _side_composition(self.ts, sites)
            states += ', transition state {}'.format(_formula(ts))
            balanced = balanced and ts == initial
        if not balanced:
            raise ValueError('{} does not conserve atoms ({}); pass '
                             'check_balance=False if this is intended'
                             ''.format(self, states))
        # charge (electrons included)
        charges = [sum(sp.charge for sp in self.reactants),
                   sum(sp.charge for sp in self.products)]
        if self.ts is not None:
            charges.append(sum(sp.charge for sp in self.ts))
        if len(set(charges)) > 1:
            names = ['reactants', 'products', 'transition state']
            raise ValueError('{} does not conserve charge ({}); pass '
                             'check_balance=False if this is intended'
                             ''.format(self, ', '.join(
                                 '{} {:+d}'.format(n, c)
                                 for n, c in zip(names, charges))))

    def _check_scale_param(self, param):
        if param not in self.scale:
            raise ValueError('{} is not a valid scaling parameter; valid '
                             'names are {}'.format(param, self.scale_params))

    def get_scale(self, param):
        """Multiplier of 'dH_act', 'dS_act', 'kfor' or 'krev'."""
        self._check_scale_param(param)
        return self.scale[param]

    def set_scale(self, param, value):
        """Set the multiplier of 'dH_act', 'dS_act', 'kfor' or 'krev' (takes
        effect at the next update)."""
        self._check_scale_param(param)
        self.scale[param] = value

    def update(self, T=None, Asite=None, L=None, force=False, U=None,
               roughness=None, delta=None):
        """Recompute the thermochemistry and rate constants at temperature
        T (K), site area Asite (m^2), diffusion length L (m), electrode
        potential U (V vs SHE), roughness factor and film thickness delta
        (for FILM), if any of them (or a scale factor) changed, or if force.
        U, roughness and delta: None keeps the current value. The Model
        calls this."""
        if not force and not self.is_update_needed(T, Asite, L, U, roughness,
                                                   delta):
            return

        if U is not None:
            self.U = U
        if roughness is not None:
            self.roughness = roughness
        if delta is not None:
            self.model_delta = delta
        for species in self.species + list(self.ts or []):
            if isinstance(species, Electron):
                species.U = self.U
        for species in self.species:
            species.update(T=T, force=True)

        self.T = T
        self.Asite = Asite
        self.L = L
        self.dH = self.products.get_H(T) - self.reactants.get_H(T)
        self.dS = self.products.get_S(T) - self.reactants.get_S(T)
        self.dG = self.dH - self.T * self.dS
        if self.method == 'FILM' and abs(_at_zero_coverage(self.dG)) > 1e-9:
            raise ValueError('FILM connects a species with its near-surface '
                             'copy, which must have the same free energy '
                             '({}: dG = {:g} eV); use species.copy()'
                             ''.format(self, _at_zero_coverage(self.dG)))
        if self.ts is not None:
            for species in self.ts:
                species.update(T=T, force=True)

            # electrochemical step: the transition state's free energy at the
            # electrode potential U from that at U_ref, so that
            # dG_act(U) = dG_act(U_ref) + beta (dG(U) - dG(U_ref))
            ts_shift = 0.
            if self.n_electrons:
                ts_shift = (-(1 - self.beta) * self.n_electrons
                            * (self.U - self.U_ref))
            Gts = self.ts.get_G(T) + ts_shift
            Gr = self.reactants.get_G(T)
            Gp = self.products.get_G(T)

            # coverage-dependent energy corrections that the TS follows
            # (weighted with alpha)
            if self.ts_follows_dE:
                dEr = np.sum([species.lateral + species.dE for species in self.reactants])
                dEp = np.sum([species.lateral + species.dE for species in self.products])
            else:
                dEr = np.sum([species.lateral for species in self.reactants])
                dEp = np.sum([species.lateral for species in self.products])

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
            self.dH_act += ts_shift
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
        elif self.dG_act0 is not None:
            # barrier dG_act0 where dG = 0, changing with beta dG (e.g.
            # "simple" PCET steps; CatMap's G_TS = G_FS + barrier
            # + (1 - beta)(-dG))
            self.dG_act = self.dG_act0 + self.beta * self.dG
        elif self.dG_act_given is not None:
            self.dG_act = self.dG_act_given
            if self.n_electrons:
                self.dG_act += (self.beta * self.n_electrons
                                * (self.U - self.U_ref))
        self._calc_keq()
        self._calc_kfor()
        self._calc_krev()
        self.scale_old = self.scale.copy()

    def is_update_needed(self, T, Asite, L, U=None, roughness=None,
                         delta=None):
        if U is not None and U != self.U:
            return True
        if roughness is not None and roughness != self.roughness:
            return True
        if delta is not None and delta is not self.model_delta \
                and delta != self.model_delta:
            return True
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

    def get_keq(self, T=None, Asite=None, L=None, U=None, roughness=None,
                delta=None):
        """Equilibrium constant (see update() for the arguments)."""
        self.update(T, Asite, L, U=U, roughness=roughness, delta=delta)
        return self.keq

    def get_kfor(self, T=None, Asite=None, L=None, U=None, roughness=None,
                delta=None):
        """Forward rate constant (see update() for the arguments)."""
        self.update(T, Asite, L, U=U, roughness=roughness, delta=delta)
        return self.kfor

    def get_krev(self, T=None, Asite=None, L=None, U=None, roughness=None,
                delta=None):
        """Reverse rate constant (see update() for the arguments)."""
        self.update(T, Asite, L, U=U, roughness=roughness, delta=delta)
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
        prefactor = (_k * self.T / _hplanck if self.prefactor is None
                     else self.prefactor)
        if self.method == 'EQUIL' and self.clip == 'coverage':
            self.kfor = prefactor * barr * self.scale['kfor']
        elif self.method == 'EQUIL':
            self.kfor = prefactor * barr * self.scale['kfor']
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
        elif self.method == 'FILM':
            # transport through a film (Nernst diffusion layer) of thickness
            # delta to the electrode: flux D (c_bulk - c_near) / delta per
            # geometric area, per site with roughness / Asite sites per area
            bulk, near = self.reactants[0], self.products[0]
            D = bulk.D if bulk.D is not None else near.D
            if D is None:
                raise ValueError('FILM needs the diffusion coefficient D of '
                                 '{}'.format(bulk))
            delta = self.delta if self.delta is not None else self.model_delta
            if delta is None:
                raise ValueError('FILM needs a film thickness: '
                                 'Reaction(..., delta=) or Model(..., '
                                 'delta=)')
            if callable(delta):
                delta = delta(D)
            self.kfor = (1000 * D * mol * self.Asite / (self.roughness * delta)
                         * barr * self.scale['kfor'])
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
            self.kfor = prefactor * barr * self.scale['kfor']
        else:
            raise ValueError("Method {} is not recognized!".format(
                self.method))
        if self.dG_reorg:
            # e.g. solvent reorganization, not in the computed barrier
            self.kfor *= np.exp(-self.dG_reorg / (kB * self.T))

    def _calc_krev(self):
        self.krev = self.kfor / self.keq

    def __repr__(self):
        string = self.reactants.__repr__() + ' <-> '
        if self.ts is not None:
            string += self.ts.__repr__() + ' <-> '
        string += self.products.__repr__()
        return string


def reactions_from_strings(species, reactions):
    """Several Reactions from strings, as a dict {name: Reaction} for
    Model.add_reactions.

    species: {label: species} (e.g. from read_from_db) or a list of
    species. reactions: {name: expression} or {name: (expression,
    {keyword arguments for Reaction})}, e.g.

        reactions_from_strings(sp, {
            'co_ads': ('co_g -> co', {'method': 'STICK'}),
            'co_ox': 'co + o <-> o-co -> co2_g',
        })

    See Reaction.from_string for the syntax.
    """
    out = OrderedDict()
    for name, spec in reactions.items():
        if isinstance(spec, str):
            expression, kwargs = spec, {}
        else:
            expression, kwargs = spec
        try:
            out[name] = Reaction.from_string(expression, species, **kwargs)
        except ValueError as e:
            raise ValueError('Reaction {}: {}'.format(name, e)) from None
    return out


class Model:
    """A microkinetic model: reactions, their species, and the DAE system
    d(y)/dt = dypdr . r(y) solved with SUNDIALS IDA.

    Parameters
    ----------
    T : float
        Temperature in K.
    Asite : float
        Area of one adsorption site in m^2 (used by STICK, ER, DIFF).
    z : float
        Diffusion length in m (DIFF rate law).
    lattice : micki.Lattice or dict, optional
        Site lattice, e.g. {slab: {slab: 6}} for one site type with 6
        neighbors; gives multi-site species their configurational entropy
        and sets the relative amounts of different site types.
    reactor : 'CSTR' or 'PFR'
        'CSTR': all concentrations are differential variables. 'PFR':
        adsorbate coverages are algebraic (pseudo steady state), so
        solve() integrates the fluid concentrations along the reactor.
    rhocat : float
        Concentration of catalytic sites (mol/L) that converts per-site
        rates into changes of fluid concentrations, d(c)/dt = rhocat *
        sum(nu r). Irrelevant for fixed fluids.
    analytic_jac : bool
        Give IDA the exact (complex-step) Jacobian instead of its
        difference-quotient approximation. The steady-state Newton solver
        always uses it.

    Changing T, Asite, z or lattice after set_initial_conditions rebuilds
    the model.

    Attributes after solving
    ------------------------
    U, r : list of dict
        Concentrations/coverages (species label -> value, including
        vacancies) and net rates (reaction name -> 1/s per site) at the
        output times.
    t : float or array
        Output times (s).
    steady_state_method : str
        Path taken by find_steady_state: 'newton', 'integrate+newton' or
        'integrate'.
    """

    def __init__(self, T, Asite, z=0, lattice=None, reactor='CSTR', rhocat=1,
                 analytic_jac=False, U_SHE=0., pH=None, roughness=1.,
                 delta=None):
        self._U_SHE = U_SHE  # electrode potential vs SHE (V)
        self.pH = pH
        self._roughness = roughness
        self._delta = delta
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

    def _conditions(self):
        """Keyword arguments for Reaction.update."""
        return {'T': self.T, 'Asite': self.Asite, 'L': self.z,
                'U': self.U_SHE, 'roughness': self.roughness,
                'delta': self.delta}

    def _set_condition(self, name, value):
        setattr(self, name, value)
        for reaction in self._reactions:
            reaction.update(**self._conditions())
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    roughness = property(
        lambda self: self._roughness,
        lambda self, value: self._set_condition('_roughness', value),
        doc='Roughness factor: electrochemical surface area / geometric area')
    delta = property(
        lambda self: self._delta,
        lambda self, value: self._set_condition('_delta', value),
        doc='Film thickness (m) for FILM transport, or a function of the '
            'diffusion coefficient (e.g. micki.electrochem.levich_delta)')

    def set_U_SHE(self, U):
        self._U_SHE = U
        for reaction in self._reactions:
            reaction.update(**self._conditions())
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_U_SHE(self):
        return self._U_SHE

    U_SHE = property(get_U_SHE, set_U_SHE,
                     doc='Electrode potential (V vs SHE)')

    def get_U_RHE(self):
        if self.pH is None:
            raise ValueError('Set Model.pH to use the RHE scale')
        return self._U_SHE + kB * self.T * np.log(10.) * self.pH

    U_RHE = property(get_U_RHE, None,
                     doc='Electrode potential (V vs RHE), read-only: '
                         'U_SHE + (kT ln 10 / e) pH')

    def set_potential(self, U, scale='SHE'):
        """Set the electrode potential U on the 'SHE' or 'RHE' scale (the
        latter needs Model.pH)."""
        if scale == 'SHE':
            self.U_SHE = U
        elif scale == 'RHE':
            if self.pH is None:
                raise ValueError('Set Model.pH to use the RHE scale')
            self.U_SHE = U - kB * self.T * np.log(10.) * self.pH
        else:
            raise ValueError("scale must be 'SHE' or 'RHE'")

    def _check_conventions(self, reactions):
        """Reactions and species of one model must have been built under the
        same conventions (micki.conventions); CatMap has no configurational
        entropy, so a lattice warns."""
        used = set()
        for reaction in reactions:
            used.add(reaction.conventions)
            for species in reaction.species + list(reaction.ts or []):
                used.add(species.conventions)
        if len(used) > 1:
            raise ValueError('Reactions and species built under different '
                             'conventions ({}) cannot be combined; see '
                             'micki.conventions'.format(
                                 ', '.join(sorted(used))))
        if 'catmap' in used and self.lattice is not None:
            warnings.warn('A lattice adds configurational entropy, which '
                          "CatMap's conventions do not have",
                          RuntimeWarning, stacklevel=3)

    def add_reactions(self, reactions):
        """Add reactions, a dict {name: Reaction}, and their species. The
        names label the rates in the results. Reactions and species built
        under different micki.conventions cannot be combined."""
        # Set up list of reactions and species
        for name, reaction in reactions.items():
            if not isinstance(reaction, Reaction):
                raise TypeError('{} is not a Reaction'.format(name))
        self._check_conventions(self._reactions + list(reactions.values()))
        for name, reaction in reactions.items():
            if reaction in self._reactions:
                continue
            self._reactions.append(reaction)
            self.reactions[name] = reaction
            for species in reaction.species:
                self._add_species(species)
            if reaction.ts is not None:
                for ts in reaction.ts:
                    ts.lattice = self.lattice
            reaction.update(**self._conditions())

    def set_solvent(self, solvent):
        """Make the Liquid species labeled solvent the solvent: its
        concentration is fixed (not a variable)."""
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
        """Keep the species with these labels (a label or a list) at their
        initial concentrations, e.g. gases at constant partial pressure."""
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
            reaction.update(**self._conditions())
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_T(self):
        return self._T

    T = property(get_T, set_T, doc='Model temperature')

    def set_Asite(self, Asite):
        self._Asite = Asite
        for reaction in self._reactions:
            reaction.update(**self._conditions())
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_Asite(self):
        return self._Asite

    Asite = property(get_Asite, set_Asite, doc='Area of an adsorption site')

    def set_z(self, z):
        self._z = z
        for reaction in self._reactions:
            reaction.update(**self._conditions())
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
        self._check_conventions(self._reactions)
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
            reaction.update(force=True, **self._conditions())
        if self.U0 is not None:
            self.set_initial_conditions(self.U0)

    def get_lattice(self):
        return self._lattice

    lattice = property(get_lattice, set_lattice, doc='Model lattice')

    def _check_sites(self):
        """Raise ValueError for adsorbates that occupy no sites although
        they are not empty sites, and (with a lattice) for transition states
        without sites, unless they are marked sitefree=True."""
        hint = ('give its sites (sites=[...]) or, if it deliberately '
                'occupies none, sitefree=True')
        for species in self._species:
            if isinstance(species, Adsorbate) and not species.ts \
                    and not species.sites and not species.sitefree:
                raise ValueError('Adsorbate {} occupies no sites and is not '
                                 'a site of any species; {}'.format(
                                     species, hint))
        if self.lattice is None:
            return
        for reaction in self._reactions:
            for ts in reaction.ts or []:
                if isinstance(ts, Adsorbate) and not ts.sites \
                        and not ts.sitefree:
                    raise ValueError(
                        'Transition state {} occupies no sites, so the '
                        'lattice gives it no configurational entropy; '
                        '{}'.format(ts, hint))

    def set_initial_conditions(self, U0):
        """Set the initial state and build the model's equations and solver.

        U0 is a dict {label: value}: fluid concentrations in M (e.g. from
        micki.utils.bar_to_molar) and adsorbate coverages (fractions of
        their site type); species not given start at 0, and empty sites
        are computed from the site balances (values given for vacancies
        are ignored). Raises ValueError if a rate expression refers to a
        species that is not in the model (e.g. through lateral
        interactions), if the sites are over-full, or if an adsorbate (or,
        with a lattice, a transition state) occupies no sites without being
        marked sitefree.
        """
        if self.initialized:
            self.finalize()

        self._check_sites()

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
            if species.label not in self.fixed + [self.solvent] \
                    and not isinstance(species, Electron):
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
            rate_for = rxn.get_kfor(**self._conditions())
            rate_rev = rxn.get_krev(**self._conditions())

            for i, species in enumerate(self._variable_species):
                rcount = rxn.reactants.species.count(species)
                pcount = rxn.products.species.count(species)
                self.dypdr[i, j] = -rcount + pcount
                if isinstance(species, _Fluid) and rxn.per_site:
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
        Newton's method, falling back to time integration (in steps of dt
        seconds, up to maxiter steps, until max |dy/dt| < epsilon) and a
        final Newton polish if Newton does not converge from the initial
        conditions. method='integrate' only integrates (also used for
        models with conserved quantities, e.g. without fixed species).
        self.steady_state_method records the path taken ('newton',
        'integrate+newton' or 'integrate').

        Returns t, U, r: the time (inf when Newton alone was used), the
        state {label: value} and the net rates {reaction name: rate}.
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
        """Integrate from the initial conditions to time t (s), with ncp
        output points. Returns U, r: lists of states and rates (dicts) at
        the output times self.t. For a PFR, t is the residence time."""
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
        """Mark the model as not initialized (set_initial_conditions
        rebuilds it)."""
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
        """Warn about rate constants above kT/h at state U (run after every
        solve)."""
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
        """A new Model with the same settings, reactions, fixed species and
        solvent (sharing the Reaction and species objects), initialized
        with the same initial conditions if initialize."""
        newmodel = Model(self.T, self.Asite, z=self.z, lattice=self.lattice,
                         reactor=self.reactor, rhocat=self.rhocat,
                         analytic_jac=self.analytic_jac,
                         U_SHE=self.U_SHE, roughness=self.roughness,
                         delta=self.delta,
                         pH=self.pH)
        newmodel.add_reactions(self.reactions)
        newmodel.set_fixed(self.fixed)
        newmodel.set_solvent(self.solvent)
        if initialize:
            newmodel.set_initial_conditions(self.U0)
        return newmodel
