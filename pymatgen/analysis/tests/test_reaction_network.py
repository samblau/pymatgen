# coding: utf-8


import os
import unittest
import time
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.util.testing import PymatgenTest
from pymatgen.analysis.reaction_network import ReactionNetwork
from pymatgen.entries.mol_entry import MoleculeEntry
from monty.serialization import loadfn

try:
    import openbabel as ob
except ImportError:
    ob = None

__author__ = "Samuel Blau"
__email__ = "samblau1@gmail.com"

test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        'test_files', 'reaction_network_files')


class TestReactionNetwork(PymatgenTest):
    @classmethod
    def setUpClass(cls):
        cls.LiEC_entries = []
        entries = loadfn(os.path.join(test_dir,"LiEC_entries.json"))
        for entry in entries:
            mol = entry["output"]["optimized_molecule"]
            E = float(entry["output"]["final_energy"])
            H = float(entry["output"]["enthalpy"])
            S = float(entry["output"]["entropy"])
            mol_entry = MoleculeEntry(molecule=mol,energy=E,enthalpy=H,entropy=S,entry_id=entry["task_id"])
            cls.LiEC_entries.append(mol_entry)

        cls.LiEC_extended_entries = []
        entries = loadfn(os.path.join(test_dir,"LiEC_extended_entries.json"))
        for entry in entries:
            mol = entry["output"]["optimized_molecule"]
            E = float(entry["output"]["final_energy"])
            H = float(entry["output"]["enthalpy"])
            S = float(entry["output"]["entropy"])
            mol_entry = MoleculeEntry(molecule=mol,energy=E,enthalpy=H,entropy=S,entry_id=entry["task_id"])
            cls.LiEC_extended_entries.append(mol_entry)

    def test_LiEC_entries(self):
        RN = ReactionNetwork(self.LiEC_entries, free_energy_cutoff=40.0)
        self.assertEqual(RN.entries_list[208].free_energy,-9522.907225166065)
        self.assertEqual(RN.entries_list[208],RN.entries["C3 H4 Li1 O3"][11][0][0])
        self.assertEqual(len(RN.entries_list),236)
        self.assertEqual(len(RN.graph.nodes),1440)
        # self.assertEqual(len(RN.graph.edges),3430)
        self.assertEqual(len(RN.graph.edges),3322)

    def test_LiEC_extended_entries(self):
        RN = ReactionNetwork(self.LiEC_extended_entries)


if __name__ == "__main__":
    unittest.main()
