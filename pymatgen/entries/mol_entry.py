# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import json

from monty.json import MontyEncoder, MontyDecoder

from pymatgen.core.composition import Composition
from monty.json import MSONable
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import metal_edge_extender

"""

"""

__author__ = "Sam Blau"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__email__ = "samblau1@gmail.com"
__status__ = "Alpha"
__date__ = "Aug 1, 2019"


class MoleculeEntry(MSONable):
    """

    """

    def __init__(self, molecule, energy, correction=0.0, enthalpy=None, entropy=None,
                 parameters=None, entry_id=None, attribute=None, mol_doc=None):
        """
        Initializes a MoleculeEntry.

        Args:
            molecule (Molecule): Molecule of interest.
            energy (float): Electronic energy of the molecule in Hartree.
            correction (float): A correction to be applied to the energy.
                This is used to modify the energy for certain analyses.
                Defaults to 0.0.
            enthalpy (float): Enthalpy of the molecule (kcal/mol). Defaults to None. 
            entropy (float): Entropy of the molecule (cal/mol.K). Defaults to None.
            parameters (dict): An optional dict of parameters associated with
                the molecule. Defaults to None.
            entry_id (obj): An optional id to uniquely identify the entry.
            attribute: Optional attribute of the entry. This can be used to
                specify that the entry is a newly found compound, or to specify
                a particular label for the entry, or else ... Used for further
                analysis and plotting purposes. An attribute can be anything
                but must be MSONable.
        """
        self.molecule = molecule
        self.uncorrected_energy = energy
        self.enthalpy = enthalpy
        self.entropy = entropy
        self.composition = molecule.composition
        self.correction = correction
        self.parameters = parameters if parameters else {}
        self.entry_id = entry_id
        self.attribute = attribute
        self.mol_doc = mol_doc if mol_doc else {}

        if self.mol_doc != {}:
            
            self.enthalpy = self.mol_doc["enthalpy_kcal/mol"]
            self.entropy = self.mol_doc["entropy_cal/molK"]
            self.entry_id = self.mol_doc["task_id"]
            if "mol_graph" in self.mol_doc:
                if isinstance(self.mol_doc["mol_graph"],MoleculeGraph):
                    self.mol_graph = self.mol_doc["mol_graph"]
                else:
                    self.mol_graph = MoleculeGraph.from_dict(self.mol_doc["mol_graph"])
            else:
                mol_graph = MoleculeGraph.with_local_env_strategy(self.molecule,
                                                                  OpenBabelNN())
                self.mol_graph = metal_edge_extender(mol_graph)
        else:
            mol_graph = MoleculeGraph.with_local_env_strategy(self.molecule,
                                                              OpenBabelNN())
            self.mol_graph = metal_edge_extender(mol_graph)

    @property
    def graph(self):
        return self.mol_graph.graph

    @property
    def edges(self):
        return self.graph.edges()

    @property
    def energy(self):
        """
        Returns the *corrected* energy of the entry.
        """
        return self.uncorrected_energy + self.correction

    @property
    def free_energy(self, temp=298.0):
        if self.enthalpy != None and self.entropy != None:
            return self.energy*27.21139+0.0433641*self.enthalpy-temp*self.entropy*0.0000433641
        else:
            return None

    @property
    def formula(self):
        return self.composition.alphabetical_formula

    @property
    def charge(self):
        return self.molecule.charge

    @property
    def Nbonds(self):
        return len(self.edges)

    def __repr__(self):
        output = ["MoleculeEntry {} - {} - E{} - C{}".format(self.entry_id,
                                                      self.formula,
                                                      self.Nbonds,
                                                      self.charge),
                  "Energy = {:.4f} Hartree".format(self.uncorrected_energy),
                  "Correction = {:.4f} Hartree".format(self.correction),
                  "Enthalpy = {:.4f} kcal/mol".format(self.enthalpy),
                  "Entropy = {:.4f} cal/mol.K".format(self.entropy),
                  "Free Energy = {:.4f} eV".format(self.free_energy),
                  "Parameters:"]
        for k, v in self.parameters.items():
            output.append("{} = {}".format(k, v))
        return "\n".join(output)

    def __str__(self):
        return self.__repr__()
