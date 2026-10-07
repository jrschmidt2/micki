### micki

A modular, extensible, robust object-oriented microkinetic modeling package
written in Python.

### DEPENDENCIES (installed automatically by pip):
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

### Installation:
#create an environment with Python >= 3.12 (e.g. with venv or uv)<br>
python3.12 -m venv ~/venvs/micki<br>
source ~/venvs/micki/bin/activate<br>

#install Micki and its dependencies<br>
pip install git+https://github.com/jrschmidt2/micki.git<br>

#or, for development, from a clone<br>
git clone https://github.com/jrschmidt2/micki.git<br>
pip install -e micki<br>

### Testing:
Run the test suite (about 15 seconds; includes a water-gas shift regression test) from the repository root:<br>
python -m unittest discover -s tests -v
