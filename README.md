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

### Installation with conda (recommended):
#fetch Micki and create a conda environment with all dependencies<br>
git clone https://github.com/jrschmidt2/micki.git<br>
cd micki<br>
conda env create -f environment.yml<br>
conda activate micki<br>

Python, numpy, sympy and ase come from conda-forge. sundials4py and Micki
itself are not on conda-forge, so `environment.yml` installs them with pip
inside the environment (Micki in editable mode, so changes in the clone take
effect immediately).

### Installation with pip:
#in an environment with Python >= 3.12<br>
pip install git+https://github.com/jrschmidt2/micki.git<br>

### Testing:
Run the test suite (about 15 seconds; includes a water-gas shift regression test) from the repository root:<br>
python -m unittest discover -s tests -v
