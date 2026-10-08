### micki

A modular, extensible, robust object-oriented microkinetic modeling package
written in Python.

### DEPENDENCIES (installed automatically, see below):
 * Python >= 3.12
 * sundials4py (the official SUNDIALS Python interface; >= 7.9, beta)
 * numpy >= 2.0
 * sympy
 * ase

No compiler, SUNDIALS build or LAPACK library is needed: sundials4py ships
prebuilt wheels that include SUNDIALS.

Earlier versions that generate Fortran and compile it with f2py are kept on
branches: `fortran-sundials7` (SUNDIALS 7 Fortran 2003 interface) and
`sundials4` (SUNDIALS 4.X FCMIX interface).

### Installation from PyPI (recommended):
#in an environment with Python >= 3.12 (e.g. a venv or a conda environment)<br>
pip install micki<br>

This installs Micki and all of its dependencies and is probably the simplest
option for most users.

### Installation with conda (for development):
#fetch Micki and create a conda environment with all dependencies<br>
git clone https://github.com/jrschmidt2/micki.git<br>
cd micki<br>
conda env create -f environment.yml<br>
conda activate micki<br>

Python, numpy, sympy and ase come from conda-forge. sundials4py and Micki
itself are not on conda-forge, so `environment.yml` installs them with pip
inside the environment (Micki in editable mode from the clone, so changes
take effect immediately).

### Testing:
Run the test suite (about 15 seconds; includes a water-gas shift regression test) from the repository root:<br>
python -m unittest discover -s tests -v

### Comparing with CatMap:
Micki's defaults follow Hermes et al., J. Chem. Phys. 151, 014112 (2019).
`micki.set_conventions('catmap')` (or `with micki.conventions('catmap'):`
around building species and reactions) switches the defaults to CatMap's:
gases at 1 bar with standard atomic weights, `clip='coverage'`, `alpha=0.5`,
transition states that follow only the lateral interactions of their
initial and final states (not `dE` shifts, which then act like energy
changes in CatMap, also in thermodynamic rate control), and a warning if a
lattice is set. Explicitly given arguments still win. The individual options
are:

| CatMap | Micki |
|---|---|
| gas free energies at 1 bar, pressures in bar | `Gas(..., pref=1)`; `micki.utils.bar_to_molar(p, T)` for concentrations |
| transition state raised to max(IS, FS, TS) at the current coverages | `Reaction(..., clip='coverage')` |
| step without a transition state (barrierless) | `method='EQUIL', clip='coverage'` |
| non-activated adsorption (collision theory prefactor) | `method='STICK', clip='coverage'` |
| first-order interactions, linear/piecewise-linear response | `micki.lateral.first_order(adsorbates, eps, response=...)` |
| transition-state interactions weighted between IS and FS | `Reaction(..., alpha=w)` (`initial_state` w=0, `intermediate_state` 0.5, `final_state` 1) |
| explicit transition-state interaction parameters | TS rows in `first_order`, `Reaction(..., explicit_ts=True)` |
| no configurational entropy | no `Model.lattice` |

`tests/catmap_wgs.py` builds a water-gas shift model under
`micki.conventions('catmap')`;
`tests/test_catmap.py` checks it against CatMap's own solution (agreement
~1e-13). CatMap itself also hard-codes CODATA-2010 kB and h (a ~5e-6
effect on rates), ignores the number of sites of
multidentate species in its site balance, and fits interaction and
transition-state energies to its descriptors unless told otherwise; for a
one-to-one comparison use the energies and interaction matrix CatMap reports.

### Releasing:
Releases are published to PyPI automatically by GitHub Actions
(`.github/workflows/publish.yml`, PyPI trusted publishing):

1. Set the new version in `pyproject.toml` and commit it.
2. Create a GitHub release with tag `v<version>` (e.g. `v2.0.0`).

Publishing the release runs the tests, builds the package and uploads it to
PyPI. The workflow refuses to publish if the tag does not match the version.
To check the build and tests without publishing, start the workflow by hand
("Run workflow" under Actions -> Publish to PyPI).
