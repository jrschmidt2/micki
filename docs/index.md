# Micki documentation

Micki is an object-oriented microkinetic modeling package in Python. You
describe species with their DFT energies and vibrational frequencies (as ASE
`Atoms` objects, ASE databases or VASP output), combine them into elementary
reactions, add the reactions to a model, and solve the resulting system of
differential-algebraic equations with SUNDIALS IDA, for the time evolution of
a CSTR or PFR or for the steady state directly.

- [User guide](user-guide.md): species, reactions, models and solvers, with a
  complete small example.
- [Reference](reference.md): every class, function, parameter and option.
- [Sensitivity analysis](analysis.md): degrees of rate control, apparent
  activation energies, reaction orders.
- [CatMap conventions](catmap.md): reproducing CatMap models, and how micki and
  CatMap compare.
- [Example: water-gas shift on Pt](examples/wgs.md): a published model with
  lateral interactions, compared with experiment.

Installation is described in the [README](../README.md).

## Theory and citation

The theory behind micki (thermochemistry of fluids and adsorbates, the lattice
partition function of multidentate adsorbates, equilibrium and rate constants,
self-consistent coverage-dependent transition states, sensitivity analysis) is
derived in

> E. D. Hermes, A. N. Janes, J. R. Schmidt, *Micki: A python-based
> object-oriented microkinetic modeling code*, J. Chem. Phys. **151**, 014112
> (2019), <https://doi.org/10.1063/1.5109116>

and in more detail in E. D. Hermes' PhD dissertation (University of
Wisconsin–Madison, 2018). Please cite the paper if you use micki. Since then,
two conventions were corrected to agree with those derivations (micki 2.1):
the free energy of fluids includes the pV = kT term, and the adsorbate symmetry
number divides the number of orientations.

## Units

| Quantity | Unit |
|---|---|
| Energies, free energies, `dE`, lateral interactions | eV |
| Entropies | eV/K |
| Vibrational frequencies | eV (as ASE uses them) |
| Temperature | K |
| Fluid (gas, liquid) concentrations | mol/L (M); `micki.utils.bar_to_molar` converts pressures |
| Adsorbate coverages | fraction of the sites of their type |
| Rates | per site per second |
| Time | s |
| Site area `Asite` | m² |
| Diffusion length `z` | m; diffusion coefficients in m²/s |

## Package layout

| Module | Contents |
|---|---|
| `micki.reactants` | `Gas`, `Liquid`, `Adsorbate`, `Electron` |
| `micki.model` | `Reaction`, `Model` |
| `micki.analysis` | `ModelAnalysis` |
| `micki.lattice` | `Lattice` |
| `micki.lateral` | `first_order` (CatMap-style lateral interactions) |
| `micki.conventions` | `set_conventions`, `conventions` |
| `micki.eref` | `EnergyReference` |
| `micki.db` | `read_from_db` (and `save_to_db` on species) |
| `micki.io` | `parse_vasp_out` |
| `micki.utils` | `bar_to_molar`, `calculate_avg_vdw_radius` |
| `micki.solver` | `IDASolver` (internal) |

The most-used names are importable from `micki` directly: `Gas`, `Liquid`,
`Adsorbate`, `Electron`, `Reaction`, `Model`, `ModelAnalysis`, `Lattice`,
`EnergyReference`, `set_conventions`, `get_conventions`, `conventions`.
