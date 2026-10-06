### micki

A modular, extensible, robust object-oriented microkinetic modeling package
written in Python.

### DEPENDENCIES:
 * Python >= 3.12
 * sundials4py (the official SUNDIALS Python interface; >= 7.9, beta)
 * numpy >= 2.0
 * sympy
 * ase

No compiler, SUNDIALS build or LAPACK library is needed: sundials4py ships
prebuilt wheels that include SUNDIALS.

Micki versions that generate and compile Fortran with the SUNDIALS FCMIX
(4.X) interface are on the `sundials4` branch.

### Detailed installation instructions:
#create an environment with Python >= 3.12 (e.g. with venv or uv) and install the dependencies<br>
python3.12 -m venv ~/venvs/micki<br>
~/venvs/micki/bin/pip install sundials4py sympy ase<br>

#fetch Micki itself and put it on the Python path<br>
git clone https://github.com/jrschmidt2/micki.git<br>
export PYTHONPATH=$PWD/micki:$PYTHONPATH<br>

### Testing:
Run the water-gas shift regression test (about 40 seconds) from the repository root:<br>
python -m unittest discover -s tests -v
