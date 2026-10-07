"""Water-gas shift microkinetic model used as a regression test.

The model (species database, energy shifts, lateral interactions, reactions)
is taken from the original wgs_tof.py example, except for the symmetry
numbers and the sites of the CO-OH transition state (see below). The 21 experimental
reaction conditions are those of Table 5 in Grabow et al., J. Phys. Chem. C
2008, 112, 4608 (conditions 13 and 17 were mistyped in wgs_tof.py).

Running this file regenerates the reference data used by test_wgs.py:

    python tests/wgs.py

Only do that after an intentional change in results, and say why in the
commit message. History of the reference: it reproduced the original
Fortran/SUNDIALS 4.X results (commit bb8aa7d) to ~1e-10, then conditions 13
and 17 were corrected (Grabow et al. Table 5), then the missing pV = kT term
in fluid free energies was added (TOFs x1.33-1.45), then the sign of the
adsorbate symmetry number was corrected (it divides the number of
orientations), symm = 2 was removed from the ho-h and o-co transition states,
and the co-oh transition state was given two sites like the other transition
states (TOFs x1.3-3.2). The energy shifts (dE) below were calibrated before
these fixes and have not been refitted.
"""

import json
import os
import sys

import numpy as np
from ase.units import _k, _Nav, m

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micki import Model, Reaction  # noqa: E402
from micki.db import read_from_db  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
REFERENCE = os.path.join(DATA_DIR, 'wgs_reference.json')

# (T [K], P_CO, P_H2O, P_CO2, P_H2 [atm], flow rate [cm3/min])
CONDITIONS = [
    (523, 0.154, 0.208, 0.000, 0.000, 102.9),  # 1
    (548, 0.055, 0.208, 0.000, 0.000, 85.9),   # 2
    (548, 0.105, 0.208, 0.000, 0.000, 97.4),   # 3
    (548, 0.137, 0.062, 0.000, 0.000, 118.0),  # 4
    (548, 0.144, 0.104, 0.000, 0.000, 114.8),  # 5
    (548, 0.148, 0.145, 0.000, 0.000, 108.4),  # 6
    (548, 0.145, 0.208, 0.000, 0.000, 101.4),  # 7
    (548, 0.106, 0.208, 0.068, 0.000, 106.9),  # 8
    (548, 0.104, 0.208, 0.109, 0.000, 105.6),  # 9
    (548, 0.140, 0.208, 0.151, 0.000, 110.1),  # 10
    (548, 0.102, 0.208, 0.192, 0.000, 109.1),  # 11
    (548, 0.134, 0.208, 0.000, 0.037, 103.5),  # 12
    (548, 0.156, 0.208, 0.000, 0.097, 102.1),  # 13
    (548, 0.130, 0.208, 0.000, 0.123, 105.9),  # 14
    (548, 0.134, 0.208, 0.177, 0.123, 95.7),   # 15
    (548, 0.132, 0.208, 0.000, 0.173, 103.8),  # 16
    (548, 0.146, 0.208, 0.000, 0.191, 94.4),   # 17
    (548, 0.159, 0.208, 0.000, 0.208, 101.1),  # 18
    (548, 0.198, 0.208, 0.000, 0.000, 102.6),  # 19
    (548, 0.223, 0.208, 0.000, 0.000, 88.3),   # 20
    (573, 0.150, 0.208, 0.000, 0.000, 103.4),  # 21
]

# number of catalyst sites: wt% * dispersion / molar mass * catalyst mass
NSITES = 47e-6 * 0.1501

# area of one site
ASITE = np.sqrt(3) * 3.8966**2 / (4 * m**2)

# initial guess for the surface coverages
COVERAGE0 = {'co': 0.56275433599205904,
             'cooh': 1.3744935762217692e-12,
             'h': 1.2798504989912461e-05,
             'h2o': 1.791655916669818e-05,
             'o': 9.4267631566852731e-16,
             'oh': 1.7137327904883546e-10}


def build_species():
    sp = read_from_db(os.path.join(DATA_DIR, 'wgs.json'),
                      eref=['slab', 'co_g', 'h2o_g', 'h2_g'])
    # The O-H-OH transition state (H transfer between two O) is end-to-end
    # symmetric. The original example also set symm = 2 for ho-h and o-co,
    # which are not.
    sp['o-h-oh'].symm = 2

    # Shift CO and H energies to match experimental binding enthalpies
    sp['co'].dE = 0.09496182099234107
    sp['h'].dE = 0.236689058 / 2.0

    # Shift energy of OH and COOH based on optimization results
    sp['oh'].dE = -0.2171335
    sp['cooh'].dE = -0.0515426

    # Lateral interactions
    sp['co'].lateral = 2 * 0.784423808 * sp['co'].symbol
    sp['o'].lateral = (1.147079243 * sp['co'].symbol
                       + 2 * 1.11913902 * sp['o'].symbol)
    sp['co'].lateral += 1.147079243 * sp['o'].symbol

    sp['h'].lateral = 0.237728186 * sp['co'].symbol
    sp['co'].lateral += 0.237728186 * sp['h'].symbol

    sp['h2o'].lateral = 0.10555879 * sp['co'].symbol
    sp['co'].lateral += 0.10555879 * sp['h2o'].symbol

    sp['oh'].lateral = 0.263160274 * sp['co'].symbol
    sp['co'].lateral += 0.263160274 * sp['oh'].symbol

    sp['cooh'].lateral = 1.901900269 * sp['co'].symbol
    sp['co'].lateral += 1.901900269 * sp['cooh'].symbol
    return sp


def build_reactions(sp):
    return {
        'co_ads': Reaction(sp['co_g'], sp['co'], method='STICK'),
        'h2o_ads': Reaction(sp['h2o_g'], sp['h2o'], method='STICK'),
        'h2_ads': Reaction(sp['h2_g'], 2 * sp['h'], method='STICK'),
        'ho-h': Reaction(sp['h2o'], sp['oh'] + sp['h'], ts=sp['ho-h']),
        'o-h': Reaction(sp['oh'], sp['o'] + sp['h'], ts=sp['o-h']),
        'o-h-oh': Reaction(2 * sp['oh'], sp['o'] + sp['h2o'],
                           method='DIEQUIL'),
        'o-co': Reaction(sp['o'] + sp['co'], sp['co2_g'], ts=sp['o-co']),
        'co-oh': Reaction(sp['cooh'], sp['co'] + sp['oh'], ts=sp['co-oh']),
        'oco-h': Reaction(sp['cooh'] + sp['slab'], sp['co2_g'] + sp['h'],
                          ts=sp['oco-h']),
        'oco-h-o': Reaction(sp['cooh'] + sp['o'], sp['co2_g'] + sp['oh'],
                            method='EQUIL'),
        'oco-h-oh': Reaction(sp['cooh'] + sp['oh'], sp['co2_g'] + sp['h2o'],
                             method='EQUIL'),
    }


def run_condition(condition, sp, rxns, analytic_jac=False):
    """Solve a CSTR to steady state, then use it to start a PFR.

    Returns the TOF, the CSTR steady state, and the PFR outlet state.
    """
    T, p_co, p_h2o, p_co2, p_h2, flow = condition
    atm2molar = 101325 / _k / T / _Nav / 1000.

    model = Model(T, ASITE, reactor='CSTR', analytic_jac=analytic_jac)
    model.lattice = {sp['slab']: {sp['slab']: 6}}
    model.add_reactions(rxns)
    model.set_fixed(['co_g', 'h2o_g', 'h2_g', 'co2_g'])
    U0 = {'co_g': atm2molar * p_co,
          'h2o_g': atm2molar * p_h2o,
          'co2_g': atm2molar * p_co2,
          'h2_g': atm2molar * p_h2}
    U0.update(COVERAGE0)
    model.set_initial_conditions(U0)
    _, U_cstr, _ = model.find_steady_state(maxiter=200000, epsilon=1e-8)

    V = 1
    dt = 60 * 1000 * NSITES / flow
    pfr = Model(T, ASITE, reactor='PFR', rhocat=1. / V,
                analytic_jac=analytic_jac)
    pfr.lattice = model.lattice
    pfr.add_reactions(rxns)
    pfr.set_initial_conditions(U_cstr)
    U_pfr, _ = pfr.solve(dt, 1000)

    tof = ((U_pfr[-1]['co2_g'] - atm2molar * p_co2) * flow
           / (1000 * NSITES))
    return (float(tof),
            {k: float(v) for k, v in U_cstr.items()},
            {k: float(v) for k, v in U_pfr[-1].items()})


def run_all(analytic_jac=False):
    sp = build_species()
    rxns = build_reactions(sp)
    results = []
    for condition in CONDITIONS:
        tof, U_cstr, U_pfr = run_condition(condition, sp, rxns, analytic_jac)
        results.append({'tof': tof, 'U_cstr': U_cstr, 'U_pfr': U_pfr})
    return results


if __name__ == '__main__':
    results = run_all()
    with open(REFERENCE, 'w') as f:
        json.dump(results, f, indent=1)
        f.write('\n')
    print('Wrote', REFERENCE)
