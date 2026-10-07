"""ModelAnalysis on the water-gas shift CSTR (condition 7).

Run from the repository root:

    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wgs  # noqa: E402
from ase.units import _k, _Nav, kB  # noqa: E402
from micki import Model, ModelAnalysis  # noqa: E402


class ModelAnalysisTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        warnings.simplefilter('ignore')
        cls.sp = wgs.build_species()
        cls.rxns = wgs.build_reactions(cls.sp)
        T, p_co, p_h2o, p_co2, p_h2, flow = wgs.CONDITIONS[6]
        atm2molar = 101325 / _k / T / _Nav / 1000.
        cls.T = T
        model = Model(T, wgs.ASITE, reactor='CSTR')
        model.lattice = {cls.sp['slab']: {cls.sp['slab']: 6}}
        model.add_reactions(cls.rxns)
        model.set_fixed(['co_g', 'h2o_g', 'h2_g', 'co2_g'])
        U0 = {'co_g': atm2molar * p_co, 'h2o_g': atm2molar * p_h2o,
              'co2_g': atm2molar * p_co2, 'h2_g': atm2molar * p_h2}
        U0.update(wgs.COVERAGE0)
        cls.model = model
        cls.analysis = ModelAnalysis(model, 'co_ads', U0)

    def _rate(self, name, dG):
        # co_ads rate at steady state with species `name` shifted by dG
        self.sp[name].dE += dG
        try:
            for rxn in self.rxns.values():
                rxn.update(T=self.T, Asite=wgs.ASITE, L=0, force=True)
            model = self.model.copy(initialize=False)
            model.set_initial_conditions(self.analysis.U)
            _, _, r = model.find_steady_state()
        finally:
            self.sp[name].dE -= dG
            for rxn in self.rxns.values():
                rxn.update(T=self.T, Asite=wgs.ASITE, L=0, force=True)
        return r['co_ads']

    def test_campbell_rate_control_sums_to_one(self):
        # all rate constants scaled together just scale the rate in a CSTR
        # with fixed gas concentrations
        total = sum(float(self.analysis.campbell_rate_control(name))
                    for name in self.rxns)
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_thermodynamic_rate_control_definition(self):
        # -(kT/r) dr/dG by a separate central difference
        for name in ['h', 'co']:
            dG = 1e-3 * kB * self.T
            r0 = self._rate(name, 0.)
            expected = -(kB * self.T / r0) * (self._rate(name, dG)
                                              - self._rate(name, -dG)) / (2 * dG)
            got = float(self.analysis.thermodynamic_rate_control(name))
            self.assertAlmostEqual(got, expected, delta=1e-4 * abs(expected),
                                   msg=name)


if __name__ == '__main__':
    unittest.main()
