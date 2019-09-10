# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import os
from os.path import join
import unittest

from pymatgen.core.structure import Molecule
from pymatgen.io.qchem.utils import map_atoms_reaction
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
        mapping = map_atoms_reaction([self.rct_1, self.rct_2], self.pro)

        self.assertDictEqual(mapping, {6: 0, 2: 1, 4: 2, 7: 3, 10: 4, 14: 5, 15: 6, 16: 7, 18: 8,
                                       3: 9, 8: 10, 0: 11, 9: 12, 5: 13, 1: 14, 11: 15, 17: 16,
                                       12: 17, 13: 18})


if __name__ == "__main__":
    unittest.main()