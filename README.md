### micki

A modular, extensible, robust object-oriented microkinetic modeling package
written in Python.

### DEPENDENCIES:
 * lapack (MKL by default "-lmkl_rt"; can specify alternative LAPACK library via MICKI_LAPACK environmental variable). Must use 32-bit integers (LP64).
 * ase
 * numpy (< 2.0; micki uses numpy.f2py.compile) and a Python with development headers (Python.h)
 * sympy
 * a Fortran compiler (e.g. gfortran)
 * sundials (C library) - Tested with version 7.9. The Fortran 2003 interface is required. Compile Sundials with the following flags to cmake: -DSUNDIALS_ENABLE_FORTRAN=ON -DSUNDIALS_ENABLE_LAPACK=ON -DSUNDIALS_INDEX_SIZE=32 -DCMAKE_POSITION_INDEPENDENT_CODE=ON

Sundials versions before 7.0 are not supported (micki uses the SUNContext/SUN_COMM_NULL API and the sundials_core library introduced in 7.0; the old FCMIX interface used through 5.x was removed in 6.0).

### Detailed installation instructions:
#install ASE (which includes numpy as a dependancy) and sympy<br>
pip3 install --user ase "numpy<2"<br>
pip3 install --user sympy<br>

#install sundials<br>
wget https://github.com/LLNL/sundials/releases/download/v7.9.0/sundials-7.9.0.tar.gz<br>
tar zxvf sundials-7.9.0.tar.gz<br>
cd sundials-7.9.0<br>
mkdir build<br>
cd build<br>
#ensure that MKL is found!<br>
. /opt/intel/oneapi/setvars.sh<br>
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/sundials -DSUNDIALS_ENABLE_FORTRAN=ON -DSUNDIALS_ENABLE_LAPACK=ON -DSUNDIALS_INDEX_SIZE=32 -DCMAKE_POSITION_INDEPENDENT_CODE=ON<br>
#make sure MKL BLAS/LAPACK were found!<br>
make<br>
make install<br>

#tell micki where sundials is installed (used to find the Fortran module files)<br>
export MICKI_SUNDIALS_DIR=$HOME/sundials<br>
export LIBRARY_PATH=$MICKI_SUNDIALS_DIR/lib64:$LIBRARY_PATH<br>
export LD_LIBRARY_PATH=$MICKI_SUNDIALS_DIR/lib64:$LD_LIBRARY_PATH<br>

#fetch Micki itself<br>
git clone https://github.com/jrschmidt2/micki.git
