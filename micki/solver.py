"""DAE solver for the microkinetic model using SUNDIALS IDA (sundials4py).

The symbolic rate expressions and their derivatives built by Model are
turned into NumPy functions with sympy.lambdify and integrated with IDA
through the official SUNDIALS Python interface.
"""

import warnings

import numpy as np
import sympy as sym

from sundials4py import core, idas

# Concentrations (and empty sites) more negative than this are treated as
# unphysical (recoverable residual error, clipped in the Jacobian).
NEG_TOL = -1e-10


class IDASolver(object):
    """Integrate M dy/dt = dypdr . r(y, vac(y)) with IDA.

    symbols: sympy symbols of the variable species (length n)
    vac_symbols: sympy symbols of the vacancies (length nvac)
    vac_exprs: vacancy concentrations as expressions of `symbols`
    rates, drdy, drdvac: reaction rates (nrxns) and their partial
        derivatives w.r.t. `symbols` (nrxns x n) and `vac_symbols`
        (nrxns x nvac)
    dypdr: stoichiometry, dy_i/dt contribution of reaction j (n x nrxns)
    dvacdy: derivative of the vacancies w.r.t. `symbols` (nvac x n)
    id_vec: 1 for differential, 0 for algebraic variables (length n)
    """

    def __init__(self, symbols, vac_symbols, vac_exprs, rates, drdy, drdvac,
                 dypdr, dvacdy, id_vec):
        self.n = len(symbols)
        self.nvac = len(vac_symbols)
        self.nrates = len(rates)
        symbols = list(symbols)
        vac_symbols = list(vac_symbols)

        self._vac = sym.lambdify([symbols], list(vac_exprs), 'numpy',
                                 cse=True)
        self._rates = sym.lambdify([symbols, vac_symbols], list(rates),
                                   'numpy', cse=True)
        self._drdy = sym.lambdify([symbols, vac_symbols],
                                  sym.Matrix(drdy), 'numpy', cse=True)
        self._drdvac = sym.lambdify([symbols, vac_symbols],
                                    sym.Matrix(drdvac), 'numpy', cse=True)

        self.dypdr = np.array(dypdr, dtype=float)
        self.dvacdy = np.array(dvacdy, dtype=float)
        self.id_vec = np.array(id_vec, dtype=float)
        self.mas = np.diag(self.id_vec)

        self._mem = None

    # Model equations

    def vacancies(self, y):
        vac = np.array(self._vac(y), dtype=float).reshape(self.nvac)
        vac[vac < NEG_TOL] = 0.
        return vac

    def rates(self, y):
        return np.array(self._rates(y, self.vacancies(y)),
                        dtype=float).reshape(self.nrates)

    def residual(self, y, yp):
        """Return the DAE residual and an error flag (1 if y < 0)."""
        ier = int(np.any(y < NEG_TOL))
        res = self.dypdr @ self.rates(y) - self.id_vec * yp
        return res, ier

    def jacobian(self, y, cj):
        """Return dF/dy + cj dF/dyp and an error flag."""
        y = np.array(y, dtype=float)
        ier = 0
        if np.any(y < NEG_TOL):
            y[y < NEG_TOL] = 0.
            ier = 1
        vac = np.array(self._vac(y), dtype=float).reshape(self.nvac)
        if np.any(vac < NEG_TOL):
            vac[vac < NEG_TOL] = 0.
            ier = 1
        drdy = np.array(self._drdy(y, vac), dtype=float).reshape(
            self.nrates, self.n)
        drdvac = np.array(self._drdvac(y, vac), dtype=float).reshape(
            self.nrates, self.nvac)
        drdy = drdy + drdvac @ self.dvacdy
        return self.dypdr @ drdy - cj * self.mas, ier

    # IDA callbacks

    def _resfn(self, t, yv, ypv, rv, user_data):
        res, ier = self.residual(core.N_VGetNumpyArray(yv),
                                 core.N_VGetNumpyArray(ypv))
        core.N_VGetNumpyArray(rv)[:] = res
        return ier

    def _jacfn(self, t, cj, yv, ypv, rv, J, user_data, tmp1, tmp2, tmp3):
        jac, ier = self.jacobian(core.N_VGetNumpyArray(yv), cj)
        core.SUNDenseMatrix_Data(J)[:, :] = jac
        return ier

    # Integration

    def _new_vector(self, values):
        v = core.N_VNew_Serial(self.n, self._ctx)
        core.N_VGetNumpyArray(v)[:] = values
        return v

    def initialize(self, y0, rtol, atol, analytic_jac=False):
        self.y0 = np.array(y0, dtype=float)
        # initial time derivatives from the rate equations
        self.yp0, _ = self.residual(self.y0, np.zeros(self.n))

        err, self._ctx = core.SUNContext_Create(core.SUN_COMM_NULL)
        self._y = self._new_vector(self.y0)
        self._yp = self._new_vector(self.yp0)
        self._atol = self._new_vector(atol)
        self._id = self._new_vector(self.id_vec)
        # all y >= 0
        self._constr = self._new_vector(np.ones(self.n))

        self._view = idas.IDACreate(self._ctx)
        self._mem = self._view.get()
        self._check(idas.IDAInit(self._mem, self._resfn, 0., self._y,
                                 self._yp), 'IDAInit')
        self._check(idas.IDASVtolerances(self._mem, rtol, self._atol),
                    'IDASVtolerances')
        self._check(idas.IDASetMaxNumSteps(self._mem, 50000),
                    'IDASetMaxNumSteps')
        self._check(idas.IDASetId(self._mem, self._id), 'IDASetId')
        self._check(idas.IDASetConstraints(self._mem, self._constr),
                    'IDASetConstraints')

        self._A = core.SUNDenseMatrix(self.n, self.n, self._ctx)
        self._LS = core.SUNLinSol_Dense(self._y, self._A, self._ctx)
        self._check(idas.IDASetLinearSolver(self._mem, self._LS, self._A),
                    'IDASetLinearSolver')
        if analytic_jac:
            self._check(idas.IDASetJacFn(self._mem, self._jacfn),
                        'IDASetJacFn')
        self.nfail = 0

    @staticmethod
    def _check(flag, name):
        if flag < 0:
            raise RuntimeError('{} failed with flag {}'.format(name, flag))

    def _step(self, tout):
        flag, tret = idas.IDASolve(self._mem, tout, self._y, self._yp,
                                   idas.IDA_NORMAL)
        if flag < 0:
            self.nfail += 1
        return (flag, tret, core.N_VGetNumpyArray(self._y).copy(),
                core.N_VGetNumpyArray(self._yp).copy())

    def find_steady_state(self, dt, maxiter, epsilon):
        """Integrate in steps of dt until max |dy/dt| < epsilon.

        Returns t, y, dy/dt and the rates at the final point.
        """
        tout = 0.
        t1 = 0.
        u1 = self.y0.copy()
        du1 = self.yp0.copy()

        idas.IDACalcIC(self._mem, idas.IDA_YA_YDP_INIT, dt)

        i = 0
        while True:
            if tout - t1 < dt * 0.01:
                tout += dt
            _, t1, u1, du1 = self._step(tout)
            i += 1
            dudt, _ = self.residual(u1, np.zeros(self.n))
            if np.max(dudt**2) < epsilon**2:
                break
            if i >= maxiter:
                warnings.warn('Steady state not reached after {} steps '
                              '(t = {})'.format(maxiter, t1),
                              RuntimeWarning, stacklevel=3)
                break
        self._warn_failures()
        return t1, u1, du1, self.rates(u1)

    def solve(self, nt, tfinal):
        """Integrate to tfinal, returning nt equally spaced points."""
        dt = tfinal / (nt - 1)
        t = np.zeros(nt)
        u = np.zeros((nt, self.n))
        du = np.zeros((nt, self.n))
        r = np.zeros((nt, self.nrates))
        u[0] = self.y0
        du[0] = self.yp0
        r[0] = self.rates(self.y0)
        tout = 0.
        for i in range(1, nt):
            tout += dt
            while tout - t[i] > dt * 0.01:
                tprev = t[i]
                flag, t[i], u[i], du[i] = self._step(tout)
                if flag < 0 and t[i] <= tprev:
                    raise RuntimeError('IDA failed (flag {}) at t = {} '
                                       'without making progress'
                                       ''.format(flag, t[i]))
            r[i] = self.rates(u[i])
        self._warn_failures()
        return t, u, du, r

    def _warn_failures(self):
        if self.nfail:
            warnings.warn('IDA reported {} failed solver call(s)'
                          ''.format(self.nfail), RuntimeWarning,
                          stacklevel=3)
            self.nfail = 0

    def num_res_evals(self):
        return idas.IDAGetNumResEvals(self._mem)[1]
