"""Water-gas shift on Pt(111): a complete Micki model, written with
reaction strings.

The model of Hermes et al., J. Chem. Phys. 151, 014112 (2019) (DFT energies
and frequencies in tests/data/wgs.json, lateral interactions, energy shifts
refitted for Micki 2.1), compared with the turnover frequencies measured by
Grabow et al., J. Phys. Chem. C 112, 4608 (2008), Table 5, at 21 reaction
conditions. For each condition, a CSTR at the inlet composition is solved to
steady state, and a plug-flow reactor is started from it and integrated over
the residence time; the TOF follows from the CO2 at the outlet.

Run from the repository root:

    python examples/wgs.py

The same model, written with species objects and operators, is the
regression test tests/wgs.py.
"""

import os
import sys

import numpy as np
from ase.units import m

REPOSITORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPOSITORY)  # use this clone's micki even if not installed

from micki import Model, reactions_from_strings  # noqa: E402
from micki.db import read_from_db  # noqa: E402
from micki.utils import bar_to_molar  # noqa: E402

DATABASE = os.path.join(REPOSITORY, 'tests', 'data', 'wgs.json')

# T (K), partial pressures of CO, H2O, CO2, H2 (atm), flow rate (cm^3/min),
# measured TOF (CO2 per site per minute)
CONDITIONS = [
    (523, 0.154, 0.208, 0.000, 0.000, 102.9, 3.68),
    (548, 0.055, 0.208, 0.000, 0.000, 85.9, 8.56),
    (548, 0.105, 0.208, 0.000, 0.000, 97.4, 8.06),
    (548, 0.137, 0.062, 0.000, 0.000, 118.0, 3.63),
    (548, 0.144, 0.104, 0.000, 0.000, 114.8, 4.81),
    (548, 0.148, 0.145, 0.000, 0.000, 108.4, 5.56),
    (548, 0.145, 0.208, 0.000, 0.000, 101.4, 7.27),
    (548, 0.106, 0.208, 0.068, 0.000, 106.9, 6.05),
    (548, 0.104, 0.208, 0.109, 0.000, 105.6, 5.59),
    (548, 0.140, 0.208, 0.151, 0.000, 110.1, 6.12),
    (548, 0.102, 0.208, 0.192, 0.000, 109.1, 6.03),
    (548, 0.134, 0.208, 0.000, 0.037, 103.5, 4.13),
    (548, 0.156, 0.208, 0.000, 0.097, 102.1, 2.77),
    (548, 0.130, 0.208, 0.000, 0.123, 105.9, 2.67),
    (548, 0.134, 0.208, 0.177, 0.123, 95.7, 2.67),
    (548, 0.132, 0.208, 0.000, 0.173, 103.8, 2.55),
    (548, 0.146, 0.208, 0.000, 0.191, 94.4, 2.28),
    (548, 0.159, 0.208, 0.000, 0.208, 101.1, 2.29),
    (548, 0.198, 0.208, 0.000, 0.000, 102.6, 7.29),
    (548, 0.223, 0.208, 0.000, 0.000, 88.3, 7.09),
    (573, 0.150, 0.208, 0.000, 0.000, 103.4, 15.44),
]

ATM = 1.01325  # bar
NSITES = 47e-6 * 0.1501  # mol of Pt sites (loading * dispersion / M * mass)
ASITE = np.sqrt(3) * 3.8966**2 / (4 * m**2)  # area of a Pt(111) site, m^2
GASES = ['co_g', 'h2o_g', 'co2_g', 'h2_g']

# a starting guess for the coverages
COVERAGE0 = {'co': 0.56, 'h': 1.3e-5, 'h2o': 1.8e-5, 'oh': 1.7e-10,
             'o': 9.4e-16, 'cooh': 1.4e-12}


def build_species():
    # All species come from the database, including the empty site 'slab'
    # (the clean Pt(111) slab, which the adsorbates list as their site).
    # The energy reference makes the slab, CO, H2O and H2 zero, so adsorbate
    # energies are binding energies.
    sp = read_from_db(DATABASE, eref=['slab', 'co_g', 'h2o_g', 'h2_g'])

    # The O-H-OH transition state is end-to-end symmetric.
    sp['o-h-oh'].symm = 2

    # CO and H binding energies shifted to experimental binding enthalpies;
    # OH and COOH shifts fitted to the measured TOFs.
    sp['co'].dE = 0.09496182099234107
    sp['h'].dE = 0.236689058 / 2
    sp['oh'].dE = -0.2280
    sp['cooh'].dE = 0.1603

    # Lateral interactions (eV per unit coverage), symmetric: CO with every
    # adsorbate, and O with itself.
    co, o = sp['co'].symbol, sp['o'].symbol
    pairs = {'co': 2 * 0.784423808, 'o': 1.147079243, 'h': 0.237728186,
             'h2o': 0.10555879, 'oh': 0.263160274, 'cooh': 1.901900269}
    sp['co'].lateral = sum(eps * sp[name].symbol
                           for name, eps in pairs.items())
    for name, eps in pairs.items():
        if name != 'co':
            sp[name].lateral = eps * co
    sp['o'].lateral += 2 * 1.11913902 * o
    return sp


def build_reactions(sp):
    # 'reactants <-> transition state -> products'; steps with a transition
    # state use transition state theory, steps without one are barrierless
    # (EQUIL) unless a method is given. Empty sites are added as needed; '*'
    # is the empty site.
    return reactions_from_strings(sp, {
        'co_ads': ('co_g + * -> co', {'method': 'STICK'}),
        'h2o_ads': ('h2o_g + * -> h2o', {'method': 'STICK'}),
        'h2_ads': ('h2_g + 2* -> 2 h', {'method': 'STICK'}),
        'ho-h': 'h2o + * <-> ho-h -> oh + h',
        'o-h': 'oh + * <-> o-h -> o + h',
        'o-h-oh': ('2 oh -> o + h2o', {'method': 'DIEQUIL'}),
        'o-co': 'o + co <-> o-co -> co2_g + 2*',
        'co-oh': 'cooh + * <-> co-oh -> co + oh',
        'oco-h': 'cooh + * <-> oco-h -> co2_g + h',
        'oco-h-o': 'cooh + o -> co2_g + oh + *',
        'oco-h-oh': 'cooh + oh -> co2_g + h2o + *',
    })


def turnover_frequency(condition, sp, reactions):
    """TOF (CO2 per site per minute) at a reaction condition."""
    T, p_co, p_h2o, p_co2, p_h2, flow, _ = condition
    inlet = {name: bar_to_molar(ATM * p, T)
             for name, p in zip(GASES, (p_co, p_h2o, p_co2, p_h2))}

    # surface at the reactor inlet: CSTR with fixed gases
    cstr = Model(T, ASITE, reactor='CSTR')
    cstr.lattice = {sp['slab']: {sp['slab']: 6}}
    cstr.add_reactions(reactions)
    cstr.set_fixed(GASES)
    cstr.set_initial_conditions({**inlet, **COVERAGE0})
    _, U_inlet, _ = cstr.find_steady_state(maxiter=200000)

    # plug-flow reactor over the residence time (s), adsorbates at
    # pseudo-steady state
    residence_time = 60 * 1000 * NSITES / flow
    pfr = Model(T, ASITE, reactor='PFR', rhocat=1.)
    pfr.lattice = cstr.lattice
    pfr.add_reactions(reactions)
    # (empty sites follow from the site balance)
    pfr.set_initial_conditions({name: value for name, value in U_inlet.items()
                                if name != 'slab'})
    U, _ = pfr.solve(residence_time, 1000)

    return (U[-1]['co2_g'] - inlet['co2_g']) * flow / (1000 * NSITES)


def main():
    sp = build_species()
    reactions = build_reactions(sp)
    print(' #   T/K  p(CO) p(H2O) p(CO2) p(H2)   TOF/min  measured')
    errors = []
    for i, condition in enumerate(CONDITIONS, 1):
        tof = turnover_frequency(condition, sp, reactions)
        T, p_co, p_h2o, p_co2, p_h2, _, measured = condition
        errors.append(np.log(tof / measured))
        print('%2d  %4d  %5.3f  %5.3f  %5.3f %5.3f  %8.2f  %8.2f'
              % (i, T, p_co, p_h2o, p_co2, p_h2, tof, measured))
    print('RMS error of ln(TOF): %.3f' % np.sqrt(np.mean(np.square(errors))))


if __name__ == '__main__':
    main()
