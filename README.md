### micki

A modular, extensible, robust object-oriented microkinetic modeling package
written in Python.

### Documentation:
See the [documentation](https://github.com/jrschmidt2/micki/blob/master/docs/index.md):
a [user guide](https://github.com/jrschmidt2/micki/blob/master/docs/user-guide.md),
a complete [reference](https://github.com/jrschmidt2/micki/blob/master/docs/reference.md)
of all classes and options, [sensitivity analysis](https://github.com/jrschmidt2/micki/blob/master/docs/analysis.md),
[CatMap conventions](https://github.com/jrschmidt2/micki/blob/master/docs/catmap.md)
and a [water-gas shift example](https://github.com/jrschmidt2/micki/blob/master/docs/examples/wgs.md).
The theory is described in E. D. Hermes, A. N. Janes, J. R. Schmidt,
J. Chem. Phys. 151, 014112 (2019), https://doi.org/10.1063/1.5109116; please
cite it if you use Micki.

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
`micki.set_conventions('catmap')` switches Micki's defaults to CatMap's
conventions (gas reference state, barrier clipping, transition-state
interactions, masses), and individual options are available too. Tests
compare Micki with CatMap's own solutions of several models (agreement
~1e-13). See [CatMap conventions](https://github.com/jrschmidt2/micki/blob/master/docs/catmap.md).

### Releasing:
Releases are published to PyPI automatically by GitHub Actions
(`.github/workflows/publish.yml`, PyPI trusted publishing):

1. Set the new version in `pyproject.toml` and commit it.
2. Create a GitHub release with tag `v<version>` (e.g. `v2.0.0`).

Publishing the release runs the tests, builds the package and uploads it to
PyPI. The workflow refuses to publish if the tag does not match the version.
To check the build and tests without publishing, start the workflow by hand
("Run workflow" under Actions -> Publish to PyPI).
