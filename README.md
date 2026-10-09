### micki

A modular, extensible, robust object-oriented microkinetic modeling package
written in Python.

### At a glance:
```python
from micki import Model, reactions_from_strings
from micki.db import read_from_db
from micki.utils import bar_to_molar

# species (energies, frequencies, sites) from an ASE database
sp = read_from_db('species.json', eref=['slab', 'co_g', 'o2_g'])

# reactions as strings: 'reactants <-> transition state -> products';
# '*' is the empty site
reactions = reactions_from_strings(sp, {
    'co_ads': ('co_g + * -> co', {'method': 'STICK'}),
    'o2_ads': 'o2_g + 2* <-> o-o -> 2 o',
    'co_ox': 'co + o <-> o-co -> co2_g + 2*',
})

T = 500.
model = Model(T, Asite=7e-20)
model.add_reactions(reactions)
model.set_fixed(['co_g', 'o2_g', 'co2_g'])
model.set_initial_conditions({'co_g': bar_to_molar(0.1, T),
                              'o2_g': bar_to_molar(0.2, T)})
t, U, r = model.find_steady_state()   # coverages U, rates r
```

### Documentation:
See the [documentation](https://github.com/jrschmidt2/micki/blob/master/docs/index.md):
a [user guide](https://github.com/jrschmidt2/micki/blob/master/docs/user-guide.md),
a complete [reference](https://github.com/jrschmidt2/micki/blob/master/docs/reference.md)
of all classes and options, [sensitivity analysis](https://github.com/jrschmidt2/micki/blob/master/docs/analysis.md),
[electrochemistry](https://github.com/jrschmidt2/micki/blob/master/docs/electrochemistry.md),
[CatMap conventions](https://github.com/jrschmidt2/micki/blob/master/docs/catmap.md)
and a [water-gas shift example](https://github.com/jrschmidt2/micki/blob/master/docs/examples/wgs.md)
(script: [examples/wgs.py](https://github.com/jrschmidt2/micki/blob/master/examples/wgs.py)).
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
~1e-13). See [electrochemistry](https://github.com/jrschmidt2/micki/blob/master/docs/electrochemistry.md),
[CatMap conventions](https://github.com/jrschmidt2/micki/blob/master/docs/catmap.md).

### Releasing:
Releases are published to PyPI automatically by GitHub Actions
(`.github/workflows/publish.yml`, PyPI trusted publishing):

1. Set the new version in `pyproject.toml` and commit it.
2. Create a GitHub release with tag `v<version>` (e.g. `v2.0.0`).

Publishing the release runs the tests, builds the package and uploads it to
PyPI. The workflow refuses to publish if the tag does not match the version.
To check the build and tests without publishing, start the workflow by hand
("Run workflow" under Actions -> Publish to PyPI).
