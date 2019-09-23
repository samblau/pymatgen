# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import os
from os.path import join
import unittest

import numpy as np

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import CovalentBondNN
from pymatgen.io.qchem.utils import (map_atoms_reaction,
                                     orient_molecule,
                                     generate_string_start)
# from pymatgen.util.testing import PymatgenTest

try:
    import openbabel
    have_babel = True
except ImportError:
    have_babel = False

__author__ = "Evan Spotte-Smith"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"

test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..",
                        'test_files', "molecules")


class QCUtilsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.rct_1 = Molecule.from_file(join(test_dir, "da_reactant_1.mol"))
        self.rct_2 = Molecule.from_file(join(test_dir, "da_reactant_2.mol"))
        self.pro = Molecule.from_file(join(test_dir, "da_product.mol"))

    def tearDown(self) -> None:
        del self.rct_1
        del self.rct_2
        del self.pro

    def test_map_atoms_reaction(self):
        rct_1 = MoleculeGraph.with_local_env_strategy(self.rct_1, CovalentBondNN(),
                                                      reorder=False, extend_structure=False)
        rct_2 = MoleculeGraph.with_local_env_strategy(self.rct_2, CovalentBondNN(),
                                                      reorder=False, extend_structure=False)
        pro = MoleculeGraph.with_local_env_strategy(self.pro, CovalentBondNN(),
                                                    reorder=False, extend_structure=False)

        mapping = map_atoms_reaction([rct_1, rct_2], pro)

        self.assertDictEqual(mapping, {6: 0, 2: 1, 4: 2, 7: 3, 10: 4, 14: 5, 15: 6, 16: 7, 18: 8,
                                       3: 9, 8: 10, 0: 11, 9: 12, 5: 13, 1: 14, 11: 15, 17: 16,
                                       12: 17, 13: 18})

    def test_orient_molecule(self):
        mol_1 = Molecule.from_file(os.path.join(test_dir, "orientation_1.mol"))
        mol_2 = Molecule.from_file(os.path.join(test_dir, "orientation_2.mol"))

        mg_1 = MoleculeGraph.with_local_env_strategy(mol_1, CovalentBondNN(),
                                                     reorder=False,
                                                     extend_structure=False)
        mg_2 = MoleculeGraph.with_local_env_strategy(mol_2, CovalentBondNN(),
                                                     reorder=False,
                                                     extend_structure=False)

        vec = orient_molecule(mg_1, mg_2)
        right_vec = [-0.99041777, -0.26688475, 0.05017607,
                     -0.0085718, -0.06095936, 0.04393405]
        for i in range(6):
            self.assertAlmostEqual(vec[i], right_vec[i], 5)

    def test_generate_string_start(self):
        strat = CovalentBondNN()
        molecules = generate_string_start([self.rct_1, self.rct_2], self.pro, strat,
                              reorder=False, extend_structure=False,
                              map_atoms=True)

        self.assertEqual(len(molecules["reactants"]), 2)
        self.assertEqual(len(molecules["products"]), 1)

        distance = np.linalg.norm(molecules["reactants"][0].center_of_mass -
                                  molecules["reactants"][1].center_of_mass)

        self.assertAlmostEqual(distance, 3.516695391504106, 5)

        molecules_large_gap = generate_string_start([self.rct_1, self.rct_2], self.pro, strat,
                              reorder=False, extend_structure=False,
                              map_atoms=True, separation_dist=2.5)

        distance_large = np.linalg.norm(molecules_large_gap["reactants"][0].center_of_mass -
                                  molecules_large_gap["reactants"][1].center_of_mass)

        self.assertAlmostEqual(distance_large, 5.516496046861798, 5)


if __name__ == "__main__":
    unittest.main()