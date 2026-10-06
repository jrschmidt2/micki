"""Regression test: water-gas shift TOFs and states for 21 conditions.

Run from the repository root (takes about a minute per test):

    python -m unittest discover -s tests -v

Requires the SUNDIALS/LAPACK environment described in README.md
(MICKI_SUNDIALS_DIR, MICKI_LAPACK, LIBRARY_PATH, LD_LIBRARY_PATH).
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wgs  # noqa: E402

# The reference matches the current code bit-for-bit on the machine that
# generated it. The tolerances leave room for different compilers and
# LAPACK builds (the solver itself uses rtol=1e-10) while still catching any
# real change in the model or solver.
TOF_RTOL = 1e-6
STATE_RTOL = 1e-6
STATE_ATOL = 1e-12


class WGSRegressionTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(wgs.REFERENCE) as f:
            cls.reference = json.load(f)

    def setUp(self):
        # Model writes solve_ida.f90 and the compiled module to the current
        # directory and imports it from there.
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        sys.path.insert(0, self._tmp.name)

    def tearDown(self):
        sys.path.remove(self._tmp.name)
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def check(self, results):
        self.assertEqual(len(results), len(self.reference))
        for i, (res, ref) in enumerate(zip(results, self.reference), 1):
            with self.subTest(condition=i):
                self.assertLessEqual(abs(res['tof'] - ref['tof']),
                                     TOF_RTOL * abs(ref['tof']),
                                     'TOF {} vs reference {}'
                                     ''.format(res['tof'], ref['tof']))
                for key in ['U_cstr', 'U_pfr']:
                    self.assertEqual(sorted(res[key]), sorted(ref[key]))
                    for name, val in ref[key].items():
                        self.assertLessEqual(
                            abs(res[key][name] - val),
                            STATE_ATOL + STATE_RTOL * abs(val),
                            '{} {}: {} vs reference {}'
                            ''.format(key, name, res[key][name], val))

    def test_difference_quotient_jacobian(self):
        self.check(wgs.run_all(analytic_jac=False))

    def test_analytic_jacobian(self):
        self.check(wgs.run_all(analytic_jac=True))


if __name__ == '__main__':
    unittest.main()
