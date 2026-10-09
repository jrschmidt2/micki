"""Electrochemistry helpers: solution species, concentrations at a given pH"""

import numpy as np
from ase.units import kB, mol

from micki.reactants import Electron, Solute

WATER_MOLARITY = 55.5  # M, pure water (reference state with activity 1)


class Electrolyte:
    """Solution species of an aqueous electrolyte on the SHE scale.

    Provides two separate things:

    1. Standard free energies of the solution species (species()). The SHE
       convention fixes the energy of H3O+ relative to H2 and the electron:
       G(H3O+, 1 M) + G(e-, 0 V vs SHE) = 1/2 G(H2, 1 bar) + G(H2O), so that
       electrode potentials are vs SHE. OH- follows from pKw and each
       conjugate base A- from its pKa (HA + H2O <-> H3O+ + A-:
       dG = kT ln(10) pKa). This makes the proton-transfer channels of all
       donors (H3O+, H2O, HA) thermodynamically consistent with each other
       and with the electrode.
    2. Concentrations at a given pH (concentrations()), from the pH alone
       (and buffer totals), for the initial conditions; no equilibrium with
       the electrode is involved.

    Parameters
    ----------
    G_H2 : float
        Free energy (eV) of H2(g) at 1 bar, in the energy reference of the
        model (e.g. 0 with H2 as the reference for H).
    G_H2O : float
        Free energy (eV) of liquid water (activity 1), e.g. 0 with H2O(l) as
        the reference for O.
    pKw : float
        Ion product of water.
    acids : dict, optional
        Weak acids, {label: {'base': label of the conjugate base, 'pKa':
        pKa, 'formula': formula of the acid, 'base_formula': formula of the
        base, 'G': free energy (eV) of the acid at 1 M (default 0),
        'charge': charge of the acid (default 0)}}, e.g. acetic acid
        {'hoac': {'base': 'oac', 'pKa': 4.76, 'formula': 'C2H4O2',
        'base_formula': 'C2H3O2'}}. Only free-energy differences between
        an acid and its base enter the proton-transfer steps.
    T : float
        Temperature (K) at which pKa and pKw apply; the resulting free
        energies are used at any temperature.
    labels : dict, optional
        Labels of the hydronium ion, water, hydroxide and the electron,
        {'proton': 'h3o_aq', 'water': 'h2o_l', 'hydroxide': 'oh_aq',
        'electron': 'e'} by default.
    """

    def __init__(self, G_H2=0., G_H2O=0., pKw=14., acids=None, T=298.15,
                 labels=None):
        self.G_H2 = G_H2
        self.G_H2O = G_H2O
        self.pKw = pKw
        self.acids = dict(acids or {})
        self.T = T
        self.labels = {'proton': 'h3o_aq', 'water': 'h2o_l',
                       'hydroxide': 'oh_aq', 'electron': 'e'}
        self.labels.update(labels or {})

    def _pK_energy(self, pK):
        return kB * self.T * np.log(10.) * pK

    def species(self):
        """{label: species}: H3O+, water (solvent, activity 1 at 55.5 M),
        OH-, the acids and their conjugate bases (Solutes) and the
        Electron."""
        L = self.labels
        G_h3o = 0.5 * self.G_H2 + self.G_H2O
        out = {
            L['proton']: Solute(L['proton'], G_h3o, 'H3O', charge=1),
            L['water']: Solute(L['water'], self.G_H2O, 'H2O',
                               rhoref=WATER_MOLARITY),
            # 2 H2O <-> H3O+ + OH-: dG = kT ln(10) pKw
            L['hydroxide']: Solute(L['hydroxide'],
                                   2 * self.G_H2O - G_h3o
                                   + self._pK_energy(self.pKw),
                                   'OH', charge=-1),
            L['electron']: Electron(L['electron']),
        }
        for acid, spec in self.acids.items():
            G_acid = spec.get('G', 0.)
            charge = spec.get('charge', 0)
            out[acid] = Solute(acid, G_acid, spec['formula'], charge=charge)
            # HA + H2O <-> H3O+ + A-: dG = kT ln(10) pKa
            out[spec['base']] = Solute(
                spec['base'], G_acid + self.G_H2O - G_h3o
                + self._pK_energy(spec['pKa']), spec['base_formula'],
                charge=charge - 1)
        return out

    def concentrations(self, pH, totals=None):
        """{label: concentration (M)} at the given pH: H3O+ (10^-pH), OH-
        (Kw / [H3O+]), water (55.5 M) and, for each acid with a total
        concentration in totals ({acid label: M}), the acid and its base
        (Henderson-Hasselbalch). Concentrations stand for activities."""
        L = self.labels
        h3o = 10. ** -pH
        out = {L['proton']: h3o,
               L['hydroxide']: 10. ** -self.pKw / h3o,
               L['water']: WATER_MOLARITY}
        for acid, total in (totals or {}).items():
            spec = self.acids[acid]
            fraction = 1. / (1. + 10. ** (pH - spec['pKa']))
            out[acid] = total * fraction
            out[spec['base']] = total * (1. - fraction)
        return out


def levich_delta(nu, rpm, D=None):
    """Nernst diffusion layer thickness (m) at a rotating disk electrode,
    delta = 1.61 D^(1/3) nu^(1/6) omega^(-1/2) (Levich), with kinematic
    viscosity nu (m^2/s, about 1e-6 for water), rotation rate rpm and
    diffusion coefficient D (m^2/s). Without D, returns the function of D
    (e.g. Model(..., delta=levich_delta(1e-6, 1600)), so that each species
    gets the thickness for its own D)."""
    omega = 2 * np.pi * rpm / 60.

    def delta(D):
        return 1.61 * D ** (1. / 3) * nu ** (1. / 6) / np.sqrt(omega)
    return delta if D is None else delta(D)


def film_rhocat(Asite, roughness, delta):
    """Concentration (M) of surface sites relative to the volume of a film
    of thickness delta (m): roughness / (Asite N_A) mol per geometric area
    divided by delta. Model(..., rhocat=film_rhocat(...)) makes the
    near-surface concentrations change at their physical rates, which
    matters when solution reactions act in the film (e.g. buffer
    equilibria) or for transients; steady states of film transport and
    surface steps alone do not depend on rhocat."""
    return roughness / (Asite * mol) / delta / 1000.


def tafel_slope(U, j, T=298.15):
    """Tafel slopes dU/dlog10|j| (mV/decade) of a polarization curve
    (potentials U in V, current densities j), by finite differences
    (numpy.gradient), and the apparent transfer coefficients
    2.303 kT / (e |slope|). Returns (slopes, coefficients) as arrays."""
    U = np.asarray(U, dtype=float)
    logj = np.log10(np.abs(np.asarray(j, dtype=float)))
    slopes = 1000 * np.gradient(U) / np.gradient(logj)
    return slopes, kB * T * np.log(10.) / np.abs(slopes / 1000)
