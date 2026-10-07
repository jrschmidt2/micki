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

# Converts a single ASE DB row to a Micki Thermo object.
def row_to_thermo(row):
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

    if thermo == 'Adsorbate':
        return Adsorbate(row.toatoms(), name, freqs,
                         ts=ts, sites=sites, dE=dE, symm=symm)
    elif thermo == 'Gas':
        return Gas(row.toatoms(), name, freqs,
                   symm=symm, spin=spin, rhoref=rhoref, dE=dE)
    elif thermo == 'Liquid':
        return Liquid(row.toatoms(), name, freqs,
                      symm=symm, spin=spin, D=D, S=S, rhoref=rhoref, dE=dE)
    else:
        raise ValueError('Unknown Thermo type {}!'.format(thermo))

# Creates a dictionary of Thermo objects from a properly-formatted ASE DB file.
def read_from_db(db, names=None, eref=None):
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
