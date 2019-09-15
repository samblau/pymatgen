# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import unittest
import os

import numpy as np
from scipy.constants import h, k, R, N_A, pi

from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.io.qchem.inputs import QCInput
from pymatgen.io.qchem.outputs import QCOutput
from pymatgen.analysis.reaction_rates import (
    ReactionRateCalculator,
    BEPRateCalculator,
    ExpandedBEPRateCalculator
)

__author__ = "Evan Spotte-Smith"
__version__ = "0.1"
__maintainer__ = "Evan Spotte-Smith"
__email__ = "espottesmith@gmail.com"
__status__ = "Alpha"
__date__ = "September 2019"

module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

fsm_outfile = os.path.join(module_dir, "..", "..", "..", "test_files", "molecules",
                           "new_qchem_files", "fsm.qout")
fsm_infile = os.path.join(module_dir, "..", "..", "..", "test_files", "molecules",
                           "new_qchem_files", "fsm.qin")


class ReactionRateCalculatorTest(unittest.TestCase):

    def setUp(self) -> None:
        infile = QCInput.from_file(fsm_infile)
        outfile = QCOutput(fsm_outfile)

        self.energies = [-497.161913928997, -533.045879937285, -1030.25766660132]
        self.enthalpies = [96.692, 79.821, 178.579]
        self.entropies = [89.627, 107.726, 140.857]

        self.rct_1 = MoleculeEntry(infile.molecule["reactants"][0], self.energies[0],
                                   enthalpy=self.enthalpies[0], entropy=self.entropies[0])
        self.rct_2 = MoleculeEntry(infile.molecule["reactants"][1], self.energies[1],
                                   enthalpy=self.enthalpies[1],
                                   entropy=self.entropies[1])
        self.pro = MoleculeEntry(infile.molecule["products"][0], self.energies[2],
                                 enthalpy=self.enthalpies[2],
                                 entropy=self.entropies[2])

        self.ts = MoleculeEntry(outfile.data["string_ts_guess"], outfile.data["string_max_energy"],
                                enthalpy=200.000, entropy=160.000)

        self.calc = ReactionRateCalculator([self.rct_1, self.rct_2], [self.pro], self.ts)

    def tearDown(self) -> None:
        del self.calc
        del self.ts
        del self.pro
        del self.rct_2
        del self.rct_1
        del self.entropies
        del self.enthalpies
        del self.energies

    def test_net_properties(self):
        self.assertAlmostEqual(self.calc.net_energy, (self.energies[2] - (self.energies[0] +
                                                                          self.energies[1])), 6)
        self.assertEqual(self.calc.net_enthalpy, (self.enthalpies[2] - (self.enthalpies[0] +
                                                                        self.enthalpies[1])))
        self.assertEqual(self.calc.net_entropy, (self.entropies[2] - (self.entropies[0] +
                                                                      self.entropies[1])))

        gibbs_300 = (self.pro.free_energy(300) - (self.rct_1.free_energy(300)
                                                  + self.rct_2.free_energy(300))) * 23.061
        self.assertEqual(self.calc.calculate_net_gibbs(300), gibbs_300)
        gibbs_100 = (self.pro.free_energy(100) - (self.rct_1.free_energy(100)
                                                  + self.rct_2.free_energy(100))) * 23.061
        self.assertEqual(self.calc.calculate_net_gibbs(100.00), gibbs_100)

        self.assertDictEqual(self.calc.calculate_net_thermo(), {"energy": self.calc.net_energy,
                                                                "enthalpy": self.calc.net_enthalpy,
                                                                "entropy": self.calc.net_entropy,
                                                                "gibbs": self.calc.calculate_net_gibbs(300.00)})

    def test_act_properties(self):
        trans_energy = self.ts.energy
        trans_enthalpy = self.ts.enthalpy
        trans_entropy = self.ts.entropy

        pro_energies = [p.energy for p in self.calc.products]
        rct_energies = [r.energy for r in self.calc.reactants]
        pro_enthalpies = [p.enthalpy for p in self.calc.products]
        rct_enthalpies = [r.enthalpy for r in self.calc.reactants]
        pro_entropies = [p.entropy for p in self.calc.products]
        rct_entropies = [r.entropy for r in self.calc.reactants]

        self.assertAlmostEqual(self.calc.calculate_act_energy(), trans_energy - sum(rct_energies), 6)
        self.assertAlmostEqual(self.calc.calculate_act_energy(reverse=True),
                               trans_energy - sum(pro_energies), 6)

        self.assertEqual(self.calc.calculate_act_enthalpy(),
                         trans_enthalpy - sum(rct_enthalpies))
        self.assertEqual(self.calc.calculate_act_enthalpy(reverse=True),
                         trans_enthalpy - sum(pro_enthalpies))

        self.assertEqual(self.calc.calculate_act_entropy(),
                         trans_entropy - sum(rct_entropies))
        self.assertEqual(self.calc.calculate_act_entropy(reverse=True),
                         trans_entropy - sum(pro_entropies))

        gibbs_300 = self.calc.calculate_act_energy() * 627.509 + self.calc.calculate_act_enthalpy() - 300 * self.calc.calculate_act_entropy() / 1000
        gibbs_300_rev = self.calc.calculate_act_energy(reverse=True) * 627.509 + self.calc.calculate_act_enthalpy(reverse=True) - 300 * self.calc.calculate_act_entropy(reverse=True) / 1000
        gibbs_100 = self.calc.calculate_act_energy() * 627.509 + self.calc.calculate_act_enthalpy() - 100 * self.calc.calculate_act_entropy() / 1000
        self.assertEqual(self.calc.calculate_act_gibbs(300), gibbs_300)
        self.assertEqual(self.calc.calculate_act_gibbs(300, reverse=True), gibbs_300_rev)
        self.assertEqual(self.calc.calculate_act_gibbs(100), gibbs_100)

        self.assertEqual(self.calc.calculate_act_thermo(temperature=300.00), {"energy": self.calc.calculate_act_energy(),
                                                                              "enthalpy": self.calc.calculate_act_enthalpy(),
                                                                              "entropy": self.calc.calculate_act_entropy(),
                                                                              "gibbs": self.calc.calculate_act_gibbs(300)})
        self.assertEqual(self.calc.calculate_act_thermo(temperature=300.00, reverse=True), {"energy": self.calc.calculate_act_energy(reverse=True),
                                                                                            "enthalpy": self.calc.calculate_act_enthalpy(reverse=True),
                                                                                            "entropy": self.calc.calculate_act_entropy(reverse=True),
                                                                                            "gibbs": self.calc.calculate_act_gibbs(300, reverse=True)})

    def test_rate_constant(self):

        gibbs_300 = self.calc.calculate_act_gibbs(300)
        gibbs_300_rev = self.calc.calculate_act_gibbs(300, reverse=True)
        gibbs_600 = self.calc.calculate_act_gibbs(600)

        # Test normal forwards and reverse behavior
        self.assertEqual(self.calc.calculate_rate_constant(),
                         k * 300 / h * np.exp(-gibbs_300 * 4184 / (R * 300)))
        self.assertEqual(self.calc.calculate_rate_constant(temperature=600),
                         k * 600 / h * np.exp(-gibbs_600 * 4184 / (R * 600)))
        self.assertEqual(self.calc.calculate_rate_constant(reverse=True),
                         k * 300 / h * np.exp(-gibbs_300_rev * 4184 / (R * 300)))

        # Test effect of kappa
        self.assertEqual(self.calc.calculate_rate_constant(),
                         self.calc.calculate_rate_constant(kappa=0.5) * 2)

    def test_rates(self):

        rate_constant = self.calc.calculate_rate_constant()
        rate_constant_600 = self.calc.calculate_rate_constant(temperature=600)
        rate_constant_rev = self.calc.calculate_rate_constant(reverse=True)
        base_rate = rate_constant

        self.assertAlmostEqual(self.calc.calculate_rate([1,1]), base_rate)
        self.assertAlmostEqual(self.calc.calculate_rate([1, 0.5]), base_rate / 2, 8)
        self.assertAlmostEqual(self.calc.calculate_rate([0.5, 1]), base_rate / 2, 8)
        self.assertAlmostEqual(self.calc.calculate_rate([0.5, 0.5]), base_rate / 4, 8)
        self.assertAlmostEqual(self.calc.calculate_rate([1], reverse=True), rate_constant_rev, 8)
        self.assertAlmostEqual(self.calc.calculate_rate([1, 1], temperature=600), rate_constant_600, 8)


class BEPReactionRateCalculatorTest(unittest.TestCase):
    def setUp(self) -> None:
        infile = QCInput.from_file(fsm_infile)

        self.energies = [-497.161913928997, -533.045879937285, -1030.25766660132]
        self.enthalpies = [96.692, 79.821, 178.579]
        self.entropies = [89.627, 107.726, 140.857]

        self.rct_1 = MoleculeEntry(infile.molecule["reactants"][0], self.energies[0],
                                   enthalpy=self.enthalpies[0], entropy=self.entropies[0])
        self.rct_2 = MoleculeEntry(infile.molecule["reactants"][1], self.energies[1],
                                   enthalpy=self.enthalpies[1],
                                   entropy=self.entropies[1])
        self.pro = MoleculeEntry(infile.molecule["products"][0], self.energies[2],
                                 enthalpy=self.enthalpies[2],
                                 entropy=self.entropies[2])

        self.calc = BEPRateCalculator([self.rct_1, self.rct_2], [self.pro], 15.0, -15.0)

    def test_act_properties(self):
        self.assertAlmostEqual(self.calc.calculate_act_energy(),
                               self.calc.ea_reference + 0.5 * (self.calc.net_enthalpy - self.calc.delta_h_reference),
                               6)
        self.assertAlmostEqual(self.calc.calculate_act_energy(reverse=True),
                               self.calc.ea_reference + 0.5 * (-1 * self.calc.net_enthalpy - self.calc.delta_h_reference),
                               6)

        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_enthalpy()
        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_entropy()
        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_gibbs(300)
        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_thermo(temperature=300.00)

    def test_rate_constant(self):
        #TODO: Test reverse? But that's already covered by test_act_properties
        rate_constant = np.exp(-self.calc.calculate_act_energy() * 4184 / (R * 300))
        rate_constant_600 = np.exp(-self.calc.calculate_act_energy() * 4184 / (R * 600))

        self.assertEqual(self.calc.calculate_rate_constant(temperature=300), rate_constant)
        self.assertEqual(self.calc.calculate_rate_constant(temperature=600), rate_constant_600)

    def test_rates(self):
        base_rate = 7.419655465973114e+20
        rate_600 = 3.9139843446679724e+29

        self.assertAlmostEqual(self.calc.calculate_rate([1, 1]) / base_rate,
                               1, 8)
        self.assertAlmostEqual(self.calc.calculate_rate([1, 0.5]) / (base_rate / 2),
                               1, 8)
        self.assertAlmostEqual(self.calc.calculate_rate([0.5, 1]) / (base_rate / 2),
                               1, 8)
        self.assertAlmostEqual(self.calc.calculate_rate([0.5, 0.5]) / (base_rate / 4),
                               1, 8)
        self.assertAlmostEqual(self.calc.calculate_rate([1, 1], kappa=0.5) / (base_rate / 2),
                               1, 8)

        self.assertAlmostEqual(self.calc.calculate_rate([1, 1], temperature=600) / rate_600, 1, 8)


class ExpandedBEPReactionRateCalculatorTest(unittest.TestCase):
    def setUp(self) -> None:
        infile = QCInput.from_file(fsm_infile)

        self.energies = [-497.161913928997, -533.045879937285, -1030.25766660132]
        self.enthalpies = [96.692, 79.821, 178.579]
        self.entropies = [89.627, 107.726, 140.857]

        self.rct_1 = MoleculeEntry(infile.molecule["reactants"][0], self.energies[0],
                                   enthalpy=self.enthalpies[0], entropy=self.entropies[0])
        self.rct_2 = MoleculeEntry(infile.molecule["reactants"][1], self.energies[1],
                                   enthalpy=self.enthalpies[1],
                                   entropy=self.entropies[1])
        self.pro = MoleculeEntry(infile.molecule["products"][0], self.energies[2],
                                 enthalpy=self.enthalpies[2],
                                 entropy=self.entropies[2])

        self.calc = ExpandedBEPRateCalculator([self.rct_1, self.rct_2], [self.pro],
                                              15.0, -0.005, -15.0, -48.0)

    def test_act_properties(self):

        delta_g_ref = self.calc.delta_e_reference * 627.509 + self.calc.delta_h_reference - 300 * self.calc.delta_s_reference / 1000
        delta_g = self.calc.calculate_net_gibbs(300)
        delta_g_rev = -delta_g

        delta_g_ref_600 = self.calc.delta_e_reference * 627.509 + self.calc.delta_h_reference - 600 * self.calc.delta_s_reference / 1000
        delta_g_600 = self.calc.calculate_net_gibbs(600)

        self.assertAlmostEqual(self.calc.calculate_act_gibbs(300),
                               self.calc.delta_ga_reference + self.calc.alpha * (delta_g - delta_g_ref))
        self.assertAlmostEqual(self.calc.calculate_act_gibbs(300, reverse=True),
                               self.calc.delta_ga_reference + self.calc.alpha * (delta_g_rev - delta_g_ref))
        self.assertAlmostEqual(self.calc.calculate_act_gibbs(600),
                               self.calc.delta_ga_reference + self.calc.alpha * (delta_g_600 - delta_g_ref_600))

        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_energy()
        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_enthalpy()
        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_entropy()
        with self.assertRaises(NotImplementedError):
            self.calc.calculate_act_thermo(temperature=300.00)

    def test_rate_constant(self):
        gibbs_300 = self.calc.calculate_act_gibbs(300)
        gibbs_600 = self.calc.calculate_act_gibbs(600)

        self.assertEqual(self.calc.calculate_rate_constant(),
                         k * 300 / h * np.exp(-gibbs_300 * 4184 / (R * 300)))
        self.assertEqual(self.calc.calculate_rate_constant(temperature=600),
                         k * 600 / h * np.exp(-gibbs_600 * 4184 / (R * 600)))

        # Test effect of kappa
        self.assertEqual(self.calc.calculate_rate_constant(),
                         self.calc.calculate_rate_constant(kappa=0.5) * 2)


if __name__ == "__main__":
    unittest.main()
