"""Reading and writing species in ASE databases"""

import warnings

from ase.db import connect
from ase.db.core import Database

from micki.reactants import Adsorbate, Gas, Liquid
from micki.eref import EnergyReference

class MickiDBReadError(ValueError):
    pass

# Attempts to parse attribute 'name' from dictionary 'data' and raises a
# parse error if it cannot be found.
def get_data(row, param):
    if param not in row.data:
        raise MickiDBReadError("DB row named {} does not have '{}' entry!"
                               "".format(row.name, param))
    return row.data[param]

def row_to_thermo(row):
    """A species (Adsorbate, Gas or Liquid) from an ASE database row written
    by save_to_db: its parameters are in row.data ('thermo', 'freqs',
    'sites', 'ts', 'symm', 'spin', 'dE', 'rhoref', 'pref', 'D', 'S'); sites
    are still labels (read_from_db resolves them)."""
    name = row.name
    freqs = get_data(row, 'freqs')
    thermo = get_data(row, 'thermo')
    sites = get_data(row, 'sites')
    rhoref = get_data(row, 'rhoref')
    dE = get_data(row, 'dE')
    symm = get_data(row, 'symm')
    ts = get_data(row, 'ts')
    spin = get_data(row, 'spin')
    D = get_data(row, 'D')
    S = get_data(row, 'S')
    pref = row.data.get('pref')  # absent in databases written before 2.1
    if pref is not None or rhoref == 1.:
        # unused with pref; else the default, which depends on
        # micki.conventions
        rhoref = None

    if thermo == 'Adsorbate':
        return Adsorbate(row.toatoms(), name, freqs,
                         ts=ts, sites=sites, dE=dE, symm=symm)
    elif thermo == 'Gas':
        return Gas(row.toatoms(), name, freqs,
                   symm=symm, spin=spin, rhoref=rhoref, dE=dE, pref=pref)
    elif thermo == 'Liquid':
        return Liquid(row.toatoms(), name, freqs,
                      symm=symm, spin=spin, D=D, S=S, rhoref=rhoref, dE=dE)
    else:
        raise ValueError('Unknown Thermo type {}!'.format(thermo))

def read_from_db(db, names=None, eref=None):
    """Read species from an ASE database (a file name or an open
    connection), as {label: species}.

    names: only return these labels. eref: labels of the species (N
    structures with N elements) that define the per-element energy
    reference (micki.EnergyReference) applied to all species, e.g. the
    clean slab and a set of gases. Rows that cannot be parsed are skipped
    with a warning.
    """
    if isinstance(db, str):
        db = connect(db)
    elif not isinstance(db, Database):
        raise ValueError("Must pass active ASE DB connection, "
                         "or name of ASE DB file!")

    species = {}

    for row in db.select():
        name = row.name
        try:
            species[name] = row_to_thermo(row)
        except MickiDBReadError:
            warnings.warn("Could not parse row {}, skipping.".format(name),
                          RuntimeWarning, stacklevel=2)

    for name, sp in species.items():
        newsites = []
        for site in sp.sites:
            if site in species:
                newsites.append(species[site])
            else:
                raise ValueError("Unknown site named {}!".format(site))
        sp.sites = newsites

    if eref is not None:
        reference = EnergyReference([species[name] for name in eref])
        for name, sp in species.items():
            sp.eref = reference

    if names is not None:
        return {name: species[name] for name in names}
    return species
