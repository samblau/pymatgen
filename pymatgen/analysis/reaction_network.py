# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import logging
import copy
import itertools
import heapq
import numpy as np
from monty.json import MSONable
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import metal_edge_extender
import networkx as nx
from networkx.algorithms import bipartite


__author__ = "Samuel Blau"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "1.0"
__maintainer__ = "Samuel Blau"
__email__ = "samblau1@gmail.com"
__status__ = "Alpha"
__date__ = "7/30/19"


logger = logging.getLogger(__name__)


class ReactionNetwork(MSONable):
    """
    Class to create a reaction network from entries

    Args:
        entries ([MoleculeEntry]): A list of ReactionNetworkEntry objects.
        electron_free_energy (float): The Gibbs free energy of an electron.
        free_energy_cutoff (float): The Gibbs free energy value above which reactions
            will not be considered during network construction. Defaults to 4.0.
        prereq_cutoff (int): The number of prerequisites at and above whih reactions
            will not be considered during network construction. Defaults to 4.
    """

    def __init__(self, entries, electron_free_energy, free_energy_cutoff=4.0, prereq_cutoff=4):

        self.free_energy_cutoff = free_energy_cutoff
        self.electron_free_energy = electron_free_energy
        self.prereq_cutoff = prereq_cutoff
        self.entries = {}
        self.entries_list = []
        self.prereq_dict = {}
        self.path_const = 500
        self.prereq_const = 50
        self.path_length_cutoff = 50

        print(len(entries),"total entries")

        get_formula = lambda x: x.formula
        get_Nbonds = lambda x: x.Nbonds
        get_charge = lambda x: x.charge

        sorted_entries_0 = sorted(entries, key=get_formula)
        for k1, g1 in itertools.groupby(sorted_entries_0, get_formula):
            sorted_entries_1 = sorted(list(g1),key=get_Nbonds)
            self.entries[k1] = {}
            for k2, g2 in itertools.groupby(sorted_entries_1, get_Nbonds):
                sorted_entries_2 = sorted(list(g2),key=get_charge)
                self.entries[k1][k2] = {}
                for k3, g3 in itertools.groupby(sorted_entries_2, get_charge):
                    sorted_entries_3 = list(g3)
                    if len(sorted_entries_3) > 1:
                        unique = []
                        for entry in sorted_entries_3:
                            isomorphic_found = False
                            for ii,Uentry in enumerate(unique):
                                if entry.mol_graph.isomorphic_to(Uentry.mol_graph):
                                    isomorphic_found = True
                                    # print("Isomorphic entries with equal charges found!")
                                    if entry.free_energy != None and Uentry.free_energy != None:
                                        if entry.free_energy < Uentry.free_energy:
                                            unique[ii] = entry
                                            if entry.energy > Uentry.energy:
                                                print("WARNING: Free energy lower but electronic energy higher!")
                                    elif entry.free_energy != None:
                                        unique[ii] = entry
                                    elif entry.energy < Uentry.energy:
                                        unique[ii] = entry
                                    break
                            if not isomorphic_found:
                                unique.append(entry)
                        self.entries[k1][k2][k3] = unique
                    else:
                        self.entries[k1][k2][k3] = sorted_entries_3
                    for entry in self.entries[k1][k2][k3]:
                        self.entries_list.append(entry)

        print(len(self.entries_list),"unique entries")

        for ii, entry in enumerate(self.entries_list):
            entry.parameters["ind"] = ii

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(len(self.entries_list)),bipartite=0)

        self.one_electron_redox()
        self.intramol_single_bond_change()
        self.intermol_single_bond_change()

    def one_electron_redox(self):
        # One electron oxidation / reduction without change to bonding
        # A^n ±e- <-> A^n±1
        # Two entries with:
        #     identical composition
        #     identical number of edges
        #     a charge difference of 1
        #     isomorphic molecule graphs

        for formula in self.entries:
            for Nbonds in self.entries[formula]:
                charges = list(self.entries[formula][Nbonds].keys())
                if len(charges) > 1:
                    for ii in range(len(charges)-1):
                        charge0 = charges[ii]
                        charge1 = charges[ii+1]
                        if charge1-charge0 == 1:
                            for entry0 in self.entries[formula][Nbonds][charge0]:
                                for entry1 in self.entries[formula][Nbonds][charge1]:
                                    if entry0.mol_graph.isomorphic_to(entry1.mol_graph):
                                        self.add_reaction([entry0],[entry1],"one_electron_redox")
                                        break

    def intramol_single_bond_change(self):
        # Intramolecular formation / breakage of one bond
        # A^n <-> B^n
        # Two entries with:
        #     identical composition
        #     number of edges differ by 1
        #     identical charge
        #     removing one of the edges in the graph with more edges yields a graph isomorphic to the other entry

        for formula in self.entries:
            Nbonds_list = list(self.entries[formula].keys())
            if len(Nbonds_list) > 1:
                for ii in range(len(Nbonds_list)-1):
                    Nbonds0 = Nbonds_list[ii]
                    Nbonds1 = Nbonds_list[ii+1]
                    if Nbonds1-Nbonds0 == 1:
                        for charge in self.entries[formula][Nbonds0]:
                            if charge in self.entries[formula][Nbonds1]:
                                for entry1 in self.entries[formula][Nbonds1][charge]:
                                    for edge in entry1.edges:
                                        mg = copy.deepcopy(entry1.mol_graph)
                                        mg.break_edge(edge[0],edge[1],allow_reverse=True)
                                        if nx.is_weakly_connected(mg.graph):
                                            for entry0 in self.entries[formula][Nbonds0][charge]:
                                                if entry0.mol_graph.isomorphic_to(mg):
                                                    self.add_reaction([entry0],[entry1],"intramol_single_bond_change")
                                                    break


    def intermol_single_bond_change(self):
        # Intermolecular formation / breakage of one bond
        # A <-> B + C aka B + C <-> A
        # Three entries with:
        #     comp(A) = comp(B) + comp(C)
        #     charge(A) = charge(B) + charge(C)
        #     removing one of the edges in A yields two disconnected subgraphs that are isomorphic to B and C

        for formula in self.entries:
            for Nbonds in self.entries[formula]:
                if Nbonds > 0:
                    for charge in self.entries[formula][Nbonds]:
                        for entry in self.entries[formula][Nbonds][charge]:
                            for edge in entry.edges:
                                bond = [(edge[0],edge[1])]
                                try:
                                    frags = entry.mol_graph.split_molecule_subgraphs(bond, allow_reverse=True)
                                    graph0 = frags[0].graph
                                    formula0 = frags[0].molecule.composition.alphabetical_formula
                                    Nbonds0 = len(frags[0].graph.edges())
                                    graph1 = frags[1].graph
                                    formula1 = frags[1].molecule.composition.alphabetical_formula
                                    Nbonds1 = len(frags[1].graph.edges())
                                    if formula0 in self.entries and formula1 in self.entries:
                                        if Nbonds0 in self.entries[formula0] and Nbonds1 in self.entries[formula1]:
                                            for charge0 in self.entries[formula0][Nbonds0]:
                                                for entry0 in self.entries[formula0][Nbonds0][charge0]:
                                                    if frags[0].isomorphic_to(entry0.mol_graph):
                                                        charge1 = charge - charge0
                                                        if charge1 in self.entries[formula1][Nbonds1]:
                                                            for entry1 in self.entries[formula1][Nbonds1][charge1]:
                                                                if frags[1].isomorphic_to(entry1.mol_graph):
                                                                    self.add_reaction([entry],[entry0,entry1],"intermol_single_bond_change")
                                                                    break
                                                        break
                                except MolGraphSplitError:
                                    pass

    def add_reaction(self,entries0,entries1,rxn_type):
        """
        Args:
            entries0 ([MoleculeEntry]): list of MoleculeEntry objects on one side of the reaction
            entries1 ([MoleculeEntry]): list of MoleculeEntry objects on the other side of the reaction
            rxn_type (string): general reaction category. At present, must be one_electron_redox or 
                              intramol_single_bond_change or intermol_single_bond_change.
        """
        if rxn_type == "one_electron_redox":
            if len(entries0) != 1 or len(entries1) != 1:
                raise RuntimeError("One electron redox requires two lists that each contain one entry!")
        elif rxn_type == "intramol_single_bond_change":
            if len(entries0) != 1 or len(entries1) != 1:
                raise RuntimeError("Intramolecular single bond change requires two lists that each contain one entry!")
        elif rxn_type == "intermol_single_bond_change":
            if len(entries0) != 1 or len(entries1) != 2:
                raise RuntimeError("Intermolecular single bond change requires two lists that contain one entry and two entries, respectively!")
        else:
            raise RuntimeError("Reaction type "+rxn_type+" is not supported!")
        if rxn_type == "one_electron_redox" or rxn_type == "intramol_single_bond_change":
            entry0 = entries0[0]
            entry1 = entries1[0]
            if rxn_type == "one_electron_redox":
                val0 = entry0.charge
                val1 = entry1.charge
                if val1<val0:
                    rxn_type_A = "One electron reduction"
                    rxn_type_B = "One electron oxidation"
                else:
                    rxn_type_A = "One electron oxidation"
                    rxn_type_B = "One electron reduction"
            elif rxn_type == "intramol_single_bond_change":
                val0 = entry0.Nbonds
                val1 = entry1.Nbonds
                if val1<val0:
                    rxn_type_A = "Intramolecular single bond breakage"
                    rxn_type_B = "Intramolecular single bond formation"
                else:
                    rxn_type_A = "Intramolecular single bond formation"
                    rxn_type_B = "Intramolecular single bond breakage"
            node_name_A = str(entry0.parameters["ind"])+","+str(entry1.parameters["ind"])
            node_name_B = str(entry1.parameters["ind"])+","+str(entry0.parameters["ind"])
            energy_A = entry1.energy-entry0.energy
            energy_B = entry0.energy-entry1.energy
            if entry1.free_energy != None and entry0.free_energy != None:
                free_energy_A = entry1.free_energy-entry0.free_energy
                free_energy_B = entry0.free_energy-entry1.free_energy
                if rxn_type == "one_electron_redox":
                    if rxn_type_A == "One electron reduction":
                        free_energy_A += -self.electron_free_energy
                        free_energy_B += self.electron_free_energy
                    else:
                        free_energy_A += self.electron_free_energy
                        free_energy_B += -self.electron_free_energy
            else:
                free_energy_A = None
                free_energy_B = None

            if free_energy_A < self.free_energy_cutoff:
                self.graph.add_node(node_name_A,rxn_type=rxn_type_A,bipartite=1,energy=energy_A,free_energy=free_energy_A)
                self.graph.add_edge(entry0.parameters["ind"],
                                    node_name_A,
                                    softplus=0.0,
                                    exponent=0.0,
                                    weight=1.0)
                self.graph.add_edge(node_name_A,
                                    entry1.parameters["ind"],
                                    softplus=self.softplus(free_energy_A),
                                    exponent=self.exponent(free_energy_A),
                                    weight=1.0)
            if free_energy_B < self.free_energy_cutoff:
                self.graph.add_node(node_name_B,rxn_type=rxn_type_B,bipartite=1,energy=energy_B,free_energy=free_energy_B)
                self.graph.add_edge(entry1.parameters["ind"],
                                    node_name_B,
                                    softplus=0.0,
                                    exponent=0.0,
                                    weight=1.0)
                self.graph.add_edge(node_name_B,
                                    entry0.parameters["ind"],
                                    softplus=self.softplus(free_energy_B),
                                    exponent=self.exponent(free_energy_B),
                                    weight=1.0)

        elif rxn_type == "intermol_single_bond_change":
            entry = entries0[0]
            entry0 = entries1[0]
            entry1 = entries1[1]
            if entry0.parameters["ind"] <= entry1.parameters["ind"]:
                two_mol_name = str(entry0.parameters["ind"])+"+"+str(entry1.parameters["ind"])
            else:
                two_mol_name = str(entry1.parameters["ind"])+"+"+str(entry0.parameters["ind"])
            node_name_A = str(entry.parameters["ind"])+","+two_mol_name
            node_name_B = two_mol_name+","+str(entry.parameters["ind"])
            rxn_type_A = "Molecular decomposition breaking one bond A->B+C"
            rxn_type_B = "Molecular formation from one new bond A+B -> C"
            energy_A = entry0.energy + entry1.energy - entry.energy
            energy_B = entry.energy - entry0.energy - entry1.energy
            if entry1.free_energy != None and entry0.free_energy != None and entry.free_energy != None:
                free_energy_A = entry0.free_energy + entry1.free_energy - entry.free_energy
                free_energy_B = entry.free_energy - entry0.free_energy - entry1.free_energy

            self.graph.add_node(node_name_A,rxn_type=rxn_type_A,bipartite=1,energy=energy_A,free_energy=free_energy_A)
            self.graph.add_node(node_name_B,rxn_type=rxn_type_B,bipartite=1,energy=energy_B,free_energy=free_energy_B)

            self.graph.add_edge(entry.parameters["ind"],
                                node_name_A,
                                softplus=self.softplus(free_energy_A),
                                exponent=self.exponent(free_energy_A),
                                weight=1.0
                                )
            self.graph.add_edge(node_name_B,
                                entry.parameters["ind"],
                                softplus=self.softplus(free_energy_B),
                                exponent=self.exponent(free_energy_B),
                                weight=1.0
                                )
            self.graph.add_edge(node_name_A,
                                entry0.parameters["ind"],
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0
                                )
            self.graph.add_edge(node_name_A,
                                entry1.parameters["ind"],
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0
                                )
            self.graph.add_edge(entry0.parameters["ind"],
                                node_name_B,
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0
                                )
            self.graph.add_edge(entry1.parameters["ind"],
                                node_name_B,
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0)

    def softplus(self,free_energy):
        return np.log(1 + (273.0 / 500.0) * np.exp(free_energy))

    def exponent(self,free_energy):
        return np.exp(free_energy)

    def characterize_path(self,path,weight):
        path_dict = {}
        path_dict["byproducts"] = []
        path_dict["prereqs"] = []
        path_dict["overall_free_energy_change"] = 0.0
        path_dict["hardest_step"] = None
        path_dict["description"] = ""
        path_dict["cost"] = 0.0

        for ii,step in enumerate(path):
            if ii != len(path)-1:
                path_dict["cost"] += self.graph.get_edge_data(step,path[ii+1])[weight]
            if ii%2 == 1:
                path_dict["overall_free_energy_change"] += self.graph.node[step]["free_energy"]
                if path_dict["description"] == "":
                    path_dict["description"] += self.graph.node[step]["rxn_type"]
                else:
                    path_dict["description"] += ", " + self.graph.node[step]["rxn_type"]
                if path_dict["hardest_step"] == None:
                    path_dict["hardest_step"] = step
                elif self.graph.node[step]["free_energy"] > self.graph.node[path_dict["hardest_step"]]["free_energy"]:
                    path_dict["hardest_step"] = step

                rxn = step.split(",")
                if "+" in rxn[0]:
                    rcts = rxn[0].split("+")
                    if rcts[0] == rcts[1]:
                        path_dict["prereqs"].append(int(rcts[0]))
                    else:
                        for rct in rcts:
                            if int(rct) != path[ii-1]:
                                path_dict["prereqs"].append(int(rct))
                elif "+" in rxn[1]:
                    prods = rxn[1].split("+")
                    if prods[0] == prods[1]:
                        path_dict["byproducts"].append(int(prods[0]))
                    else:
                        for prod in prods:
                            if int(prod) != path[ii+1]:
                                path_dict["byproducts"].append(int(prod))

        path_dict["path"] = path
        path_dict["hardest_step_deltaG"] = self.graph.node[path_dict["hardest_step"]]["free_energy"]
        return path_dict

    def join_paths(self,pr_path_dict,orig_path_dict):
        path_dict = {}
        if pr_path_dict["path"][-1] not in orig_path_dict["prereqs"]:
            raise RuntimeError("Prereq path product must be a prereq of the original path!")
        else:
            easy_adds = ["path","prereqs","byproducts","overall_free_energy_change","cost"]
            for val in easy_adds:
                path_dict[val] = pr_path_dict[val] + orig_path_dict[val]
            path_dict["description"] = pr_path_dict["description"] + " to get prerequisite " + str(pr_path_dict["path"][-1]) + ". Then, " + orig_path_dict["description"]
            path_dict["prereqs"].remove(pr_path_dict["path"][-1])
            if pr_path_dict["hardest_step_deltaG"] > orig_path_dict["hardest_step_deltaG"]:
                path_dict["hardest_step_deltaG"] = pr_path_dict["hardest_step_deltaG"]
                path_dict["hardest_step"] = pr_path_dict["hardest_step"]
            else:
                path_dict["hardest_step_deltaG"] = orig_path_dict["hardest_step_deltaG"]
                path_dict["hardest_step"] = orig_path_dict["hardest_step"]
            return path_dict

    def valid_shortest_simple_paths(self,start,target,weight):
        valid_graph = copy.deepcopy(self.graph)
        to_remove = []
        for node in valid_graph.nodes():
            if valid_graph.node[node]["bipartite"] == 1:
                if "+" in node.split(",")[0]:
                    if str(target) in node.split(",")[0].split("+"):
                        to_remove.append(node)
        valid_graph.remove_nodes_from(to_remove)
        return nx.shortest_simple_paths(valid_graph,hash(start),hash(target),weight=weight)

    def find_paths(self,starts,target,weight,num_paths=10):
        """
        Args:
            starts ([int]): List of starting node IDs (ints). 
            target (int): Target node ID.
            weight (str): String identifying what edge weight to use for path finding.
            num_paths (int): Number of paths to find. Defaults to 10.
        """
        no_pr_paths = []
        c = itertools.count()
        my_heapq = []

        if weight not in self.prereq_dict:
            self.prereq_dict[weight] = {}

        for start in starts:
            ind = 0
            for path in self.valid_shortest_simple_paths(start,target,weight):
                if ind == self.path_const:
                    break
                else:
                    ind += 1
                    path_dict = self.characterize_path(path,weight=weight)
                    if len(path_dict["prereqs"]) < self.prereq_cutoff:
                        heapq.heappush(my_heapq, (path_dict["cost"],next(c),path_dict))

        while len(no_pr_paths) < num_paths and my_heapq:
            (cost, _, path_dict) = heapq.heappop(my_heapq)
            print(len(no_pr_paths),cost,len(my_heapq),path_dict["prereqs"])
            if len(path_dict["prereqs"]) == 0:
                no_pr_paths.append(path_dict)
            else:
                prereq = path_dict["prereqs"][0]
                if prereq in path_dict["byproducts"]:
                    path_dict["byproducts"].remove(prereq)
                    path_dict["prereqs"].remove(prereq)
                    heapq.heappush(my_heapq, (path_dict["cost"],next(c),path_dict))
                else:
                    for start in starts:
                        if (start,prereq) in self.prereq_dict[weight]:
                            for pr_path_dict in self.prereq_dict[weight][(start,prereq)]:
                                new_dict = self.join_paths(pr_path_dict,path_dict)
                                if len(new_dict["prereqs"]) < self.prereq_cutoff and len(new_dict["path"]) < self.path_length_cutoff:
                                    heapq.heappush(my_heapq, (new_dict["cost"],next(c),new_dict))
                        else:
                            self.prereq_dict[weight][(start,prereq)] = []
                            ind = 0
                            for path in self.valid_shortest_simple_paths(start,prereq,weight):
                                if ind == self.prereq_const:
                                    break
                                else:
                                    ind += 1
                                    pr_path_dict = self.characterize_path(path,weight=weight)
                                    self.prereq_dict[weight][(start,prereq)].append(pr_path_dict)
                                    new_dict = self.join_paths(pr_path_dict,path_dict)
                                    if len(new_dict["prereqs"]) < self.prereq_cutoff and len(new_dict["path"]) < self.path_length_cutoff:
                                        heapq.heappush(my_heapq, (new_dict["cost"],next(c),new_dict))
        return no_pr_paths

    def identify_sinks(self):
        sinks = []
        for node in self.graph.nodes():
            if self.graph.node[node]["bipartite"] == 0:
                neg_found = False
                for neighbor in list(self.graph.neighbors(node)):
                    if self.graph.node[neighbor]["free_energy"] < 0:
                        neg_found = True
                        break
                if not neg_found:
                    sinks.append(node)
        return sinks







