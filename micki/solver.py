"""DAE solver for the microkinetic model using SUNDIALS IDA (sundials4py).

The symbolic rate expressions and their derivatives built by Model are
turned into NumPy functions with sympy.lambdify and integrated with IDA
through the official SUNDIALS Python interface.
"""

import math
import warnings

import numpy as np
import sympy as sym

from sundials4py import core, idas

# Concentrations (and empty sites) more negative than this are treated as
# unphysical (recoverable residual error, clipped in the Jacobian).
NEG_TOL = -1e-10


def _complex_step_abs(z):
    # |z| for real z that keeps the derivative under complex-step
    # differentiation (numpy.abs of a complex number is its modulus, which
    # drops the imaginary perturbation)
    return np.where(np.real(z) < 0, -z, z)


# Functions used when lambdifying the rate expressions. Piecewise, Max, Min
# and Heaviside already work with complex-step differentiation (numpy
# compares complex numbers by their real part first); Abs needs the
# replacement above. re, im, conjugate and arg are not supported.
_LAMBDIFY_MODULES = [{'Abs': _complex_step_abs}, 'numpy']


# Replacements for the fast math-module functions so that they give the same
# results as the numpy versions for nan arguments (Python's max/min depend on
# argument order then) and fall back to numpy for complex intermediates
# (numpy gives nan, Python's abs the modulus). Heaviside needs nothing: both
# printers map nan to 1.
def _math_max(*args):
    return math.nan if any(a != a for a in args) else max(args)


def _math_min(*args):
    return math.nan if any(a != a for a in args) else min(args)


def _math_abs(x):
    if isinstance(x, complex):
        raise TypeError('complex value')
    return abs(x)


_MATH_MODULES = [{'Max': _math_max, 'Min': _math_min, 'Abs': _math_abs},
                 'math']


class IDASolver(object):
    """Integrate M dy/dt = dypdr . r(y, vac(y), c) with IDA.

    symbols: sympy symbols of the variable species (length n)
    vac_symbols: sympy symbols of the vacancies (length nvac)
    vac_exprs: vacancy concentrations as expressions of `symbols` and
        `fixed_symbols`
    rates: reaction rates (nrxns) as expressions of `symbols`,
        `vac_symbols` and `fixed_symbols`
    dypdr: stoichiometry, dy_i/dt contribution of reaction j (n x nrxns)
    id_vec: 1 for differential, 0 for algebraic variables (length n)
    fixed_symbols, fixed_values: species held at constant concentration c
        (fixed species and solvent); their values are passed at evaluation
        time rather than substituted into the expressions, which is much
        faster to set up.

    The Jacobian is computed by complex-step differentiation of the rate
    functions, which is exact to rounding (no symbolic derivatives needed).
    """

    # complex-step size; any tiny value gives derivatives exact to rounding
    _H = 1e-30

    def __init__(self, symbols, vac_symbols, vac_exprs, rates, dypdr, id_vec,
                 fixed_symbols=(), fixed_values=()):
        self.n = len(symbols)
        self.nvac = len(vac_symbols)
        self.nrates = len(rates)
        symbols = list(symbols)
        vac_symbols = list(vac_symbols)
        fixed_symbols = list(fixed_symbols)
        self.fixed_values = np.array(fixed_values, dtype=float)

        self._vac = sym.lambdify([symbols, fixed_symbols], list(vac_exprs),
                                 _LAMBDIFY_MODULES, cse=True)
        self._rates = sym.lambdify([symbols, vac_symbols, fixed_symbols],
                                   list(rates), _LAMBDIFY_MODULES, cse=True)
        # Versions on plain Python floats with the math module, ~4x faster
        # for the real-valued evaluations that dominate integration (numpy
        # has a large per-operation overhead on scalars). Checked against
        # the numpy versions in initialize().
        self._fixed_list = self.fixed_values.tolist()
        try:
            self._vac_math = sym.lambdify([symbols, fixed_symbols],
                                          list(vac_exprs), _MATH_MODULES,
                                          cse=True)
            self._rates_math = sym.lambdify(
                [symbols, vac_symbols, fixed_symbols], list(rates),
                _MATH_MODULES, cse=True)
            self._fast = True
        except Exception:
            self._fast = False

        self.dypdr = np.array(dypdr, dtype=float)
        self.id_vec = np.array(id_vec, dtype=float)
        self.mas = np.diag(self.id_vec)
        # Conserved quantities (e.g. element balances when no species are
        # fixed) make the steady-state equations singular; newton() is not
        # used then.
        self.has_conservation = (np.linalg.matrix_rank(self.dypdr)
                                 < self.n)

        self._mem = None

    # Model equations

    @staticmethod
    def _to_array(values, size, shape, dtype):
        # lambdify returns a list with one entry per expression; entries
        # that do not depend on the arguments are scalars
        if not shape:
            return np.array(values, dtype=dtype).reshape(size)
        out = np.empty((size,) + shape, dtype=dtype)
        for i, v in enumerate(values):
            out[i] = v
        return out

    def _raw_vacancies(self, y, shape, dtype):
        return self._to_array(self._vac(y, self.fixed_values), self.nvac,
                              shape, dtype)

    def _raw_rates(self, y, vac, shape, dtype):
        return self._to_array(self._rates(y, vac, self.fixed_values),
                              self.nrates, shape, dtype)

    # Exceptions the math-module versions raise where numpy would return
    # inf/nan (overflow, domain errors, complex results of powers); such
    # calls are repeated with the numpy versions.
    _MATH_ERRORS = (ArithmeticError, ValueError, TypeError)

    def _fast_vacancies(self, yl):
        return [0. if v < NEG_TOL else v
                for v in self._vac_math(yl, self._fixed_list)]

    def vacancies(self, y):
        if self._fast:
            try:
                return np.array(self._fast_vacancies(np.asarray(y).tolist()),
                                dtype=float).reshape(self.nvac)
            except self._MATH_ERRORS:
                pass
        vac = self._raw_vacancies(y, (), float)
        vac[vac < NEG_TOL] = 0.
        return vac

    def rates(self, y):
        if self._fast:
            try:
                yl = np.asarray(y).tolist()
                return np.array(self._rates_math(yl, self._fast_vacancies(yl),
                                                 self._fixed_list),
                                dtype=float).reshape(self.nrates)
            except self._MATH_ERRORS:
                pass
        vac = self._raw_vacancies(y, (), float)
        vac[vac < NEG_TOL] = 0.
        return self._raw_rates(y, vac, (), float)

    def residual(self, y, yp):
        """Return the DAE residual and an error flag (1 if y < 0)."""
        ier = int(np.min(y) < NEG_TOL) if self.n else 0
        res = self.dypdr @ self.rates(y) - self.id_vec * yp
        return res, ier

    def _check_fast_path(self, y):
        # use the math-module functions only if they agree with the numpy
        # ones (e.g. a function without a math equivalent fails here)
        if not self._fast:
            return
        try:
            fast = self.rates(y)
            self._fast = False
            slow = self.rates(y)
            self._fast = bool(np.allclose(fast, slow, rtol=1e-12, atol=0.))
        except Exception:
            self._fast = False

    def jacobian(self, y, cj):
        """Return dF/dy + cj dF/dyp and an error flag.

        Negative concentrations and vacancies (below NEG_TOL) are clipped to
        zero, keeping their derivatives, and flagged as a recoverable error.
        """
        y = np.array(y, dtype=float)
        ier = 0
        if np.any(y < NEG_TOL):
            y[y < NEG_TOL] = 0.
            ier = 1
        # column j of Y is y perturbed by i*H in component j; evaluating all
        # columns at once gives the whole Jacobian in one function call
        Y = y[:, None] + 1j * self._H * np.eye(self.n)
        vac = self._raw_vacancies(Y, (self.n,), complex)
        neg = vac.real < NEG_TOL
        if np.any(neg):
            vac[neg] = 1j * vac[neg].imag
            ier = 1
        r = self._raw_rates(Y, vac, (self.n,), complex)
        return (self.dypdr @ r).imag / self._H - cj * self.mas, ier

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
        self._check_fast_path(self.y0)
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

    def find_steady_state(self, dt, maxiter, epsilon, method='hybrid'):
        """Find the steady state, starting from the initial conditions.

        method='hybrid' (default) solves the steady-state equations directly
        with Newton's method (see newton()); if that fails, it integrates
        in time (integrate_to_steady_state) and then polishes the result
        with Newton. method='integrate' only integrates; it is also used when
        the model has conserved quantities (has_conservation).

        Returns t, y, dy/dt and the rates at the steady state. t is inf if
        the steady state was found by Newton alone. self.steady_state_method
        records which path was taken: 'newton', 'integrate+newton' or
        'integrate'.
        """
        if method not in ('hybrid', 'integrate'):
            raise ValueError("method must be 'hybrid' or 'integrate'")
        if self.has_conservation:
            method = 'integrate'
        if method == 'hybrid':
            y, converged = self.newton(self.y0)
            if converged:
                self.steady_state_method = 'newton'
                dydt, _ = self.residual(y, np.zeros(self.n))
                return np.inf, y, dydt, self.rates(y)
        t, y, dydt, r = self.integrate_to_steady_state(dt, maxiter, epsilon)
        self.steady_state_method = 'integrate'
        if method == 'hybrid':
            y_polished, converged = self.newton(y)
            if converged:
                self.steady_state_method = 'integrate+newton'
                dydt, _ = self.residual(y_polished, np.zeros(self.n))
                return t, y_polished, dydt, self.rates(y_polished)
        return t, y, dydt, r

    def newton(self, y_guess, xtol=1e-10, maxiter=50, floor=1e-30):
        """Solve the steady-state equations dypdr . r(y) = 0 directly.

        Newton's method on x = ln(y), which keeps all concentrations
        positive and makes the problem well scaled even when they span many
        orders of magnitude, with a backtracking line search on the
        residual scaled by the gross flux through each species. Trial
        points with negative vacancies are rejected. Converged when the
        full Newton step changes every concentration by less than xtol
        (relative); this is not limited by the rounding noise in the net
        rates the way a test on |dy/dt| is.

        Newton only converges from a reasonably close guess. Returns
        (y, converged); a singular Jacobian also gives converged=False.
        Not valid when there are conserved quantities (has_conservation),
        since nothing constrains them.
        """
        def evaluate(x):
            y = np.exp(x)
            if self.nvac and np.any(self._raw_vacancies(y, (), float) < 0):
                return None
            F, _ = self.residual(y, np.zeros(self.n))
            J, _ = self.jacobian(y, 0.)
            Jx = J * y                     # derivative w.r.t. ln(y)
            # ~ gross flux through each species
            scale = np.abs(Jx).max(axis=1)
            if not np.all(np.isfinite(F)) or not np.all(scale > 0):
                return None
            return F, Jx, scale

        x = np.log(np.maximum(np.array(y_guess, dtype=float), floor))
        state = evaluate(x)
        if state is None:
            return np.exp(x), False
        F, Jx, scale = state
        for _ in range(maxiter):
            try:
                dx = np.linalg.solve(Jx, -F)
            except np.linalg.LinAlgError:
                return np.exp(x), False
            if not np.all(np.isfinite(dx)):
                return np.exp(x), False
            if np.abs(dx).max() < xtol:
                return np.exp(x + dx), True
            # limit any single change to a factor of ~150 far from the
            # solution, then backtrack until the scaled residual decreases
            dx *= min(1., 5. / np.abs(dx).max())
            phi = np.linalg.norm(F / scale)
            lam = 1.
            while lam > 1e-6:
                trial = evaluate(x + lam * dx)
                if trial is not None and (np.linalg.norm(trial[0] / scale)
                                          < (1 - 1e-4 * lam) * phi):
                    break
                lam *= 0.5
            else:
                return np.exp(x), False
            x = x + lam * dx
            F, Jx, scale = trial
        return np.exp(x), False

    def integrate_to_steady_state(self, dt, maxiter, epsilon):
        """Integrate in steps of dt until max |dy/dt| < epsilon.

        dy/dt is taken from IDA's solution (y'), for the differential
        variables only. Re-evaluating it from the rate expressions instead
        is limited by cancellation between large forward and reverse fluxes
        (rounding noise ~1e-8-1e-7 for typical surface kinetics), which can
        keep the test from ever passing at steady state.

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
            if np.max(np.abs(du1 * self.id_vec)) < epsilon:
                break
            if i >= maxiter:
                warnings.warn('Steady state not reached after {} steps '
                              '(t = {})'.format(maxiter, t1),
                              RuntimeWarning, stacklevel=4)
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
