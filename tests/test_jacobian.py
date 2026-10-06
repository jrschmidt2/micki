"""IDASolver: complex-step Jacobian with non-analytic functions, fast path.

Run from the repository root:

    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

import numpy as np
import sympy as sym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micki.solver import IDASolver  # noqa: E402


class ComplexStepJacobianTest(unittest.TestCase):

    def setUp(self):
        a, b, v, g = sym.symbols('a b v g')
        self.symbols = [a, b]
        # one vacancy, one fixed species
        rates = [
            2.0 * g * v * (1 + sym.Abs(a - 0.3)) - 3.0 * a,
            sym.Piecewise((a * b, a < 0.4), (2 * a * b**2, True)),
            sym.Max(a, b) * v,
            sym.Min(a**2, 0.05) * sym.exp(b),
            sym.Heaviside(b - 0.2) * a * b,
        ]
        self.rates = rates
        dypdr = [[1, -1, 0, -2, 1],
                 [0, 1, -1, 1, -1]]
        self.solver = IDASolver(self.symbols, [v], [1 - a - b], rates,
                                dypdr, [1., 1.], [g], [0.7])
        self.subs_extra = {g: 0.7}
        self.v = v
        # points away from all breakpoints (a = 0.3, a = 0.4, a = b,
        # a**2 = 0.05, b = 0.2)
        self.points = [(0.15, 0.10), (0.35, 0.30), (0.50, 0.12), (0.60, 0.35)]

    def test_rates_match_sympy(self):
        a, b = self.symbols
        for y in self.points:
            subs = {a: y[0], b: y[1], self.v: 1 - y[0] - y[1]}
            subs.update(self.subs_extra)
            expected = [float(r.subs(subs)) for r in self.rates]
            np.testing.assert_allclose(self.solver.rates(np.array(y)),
                                       expected, rtol=1e-14)

    def test_jacobian_matches_finite_differences(self):
        s = self.solver
        for y in self.points:
            y = np.array(y)
            for cj in (0., 2.5):
                J, ier = s.jacobian(y, cj)
                self.assertEqual(ier, 0)
                Jfd = np.empty((s.n, s.n))
                h = 1e-6
                for j in range(s.n):
                    e = np.zeros(s.n)
                    e[j] = h
                    Jfd[:, j] = (s.residual(y + e, np.zeros(s.n))[0]
                                 - s.residual(y - e, np.zeros(s.n))[0]) / (2 * h)
                Jfd -= cj * s.mas
                np.testing.assert_allclose(J, Jfd, rtol=1e-7, atol=1e-8,
                                           err_msg='y = {}'.format(y))


class FastPathTest(unittest.TestCase):
    """The math-module evaluation path and its fallback to numpy."""

    def test_overflow_falls_back_to_numpy(self):
        a, b, v = sym.symbols('a b v')
        rates = [sym.exp(800 * a) * b, a - b]
        s = IDASolver([a, b], [v], [1 - a - b], rates, [[1, -1], [-1, 1]],
                      [1., 1.])
        s.initialize([0.1, 0.2], 1e-10, np.array([1e-16, 1e-16]))
        self.assertTrue(s._fast)
        # exp(800) overflows: math raises OverflowError, numpy gives inf
        with np.errstate(over='ignore', invalid='ignore'):
            r = s.rates(np.array([1.0, 0.0]))
        self.assertTrue(np.isnan(r[0]) or np.isinf(r[0]))
        self.assertEqual(r[1], 1.0)

    def test_fast_and_numpy_paths_agree(self):
        s = ComplexStepJacobianTest('setUp')
        s.setUp()
        solver = s.solver
        solver.initialize([0.35, 0.3], 1e-10, np.array([1e-16, 1e-16]))
        for y in s.points:
            y = np.array(y)
            solver._fast = True
            fast = solver.residual(y, np.zeros(2))[0]
            solver._fast = False
            slow = solver.residual(y, np.zeros(2))[0]
            np.testing.assert_allclose(fast, slow, rtol=1e-14)


if __name__ == '__main__':
    unittest.main()
