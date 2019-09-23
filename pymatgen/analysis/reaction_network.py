# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


import logging
import copy
import itertools
import heapq
import numpy as np
from monty.json import MSONable, MontyDecoder
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import metal_edge_extender
import networkx as nx
from networkx.algorithms import bipartite
from pymatgen.entries.mol_entry import MoleculeEntry


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
        input_entries ([MoleculeEntry]): A list of MoleculeEntry objects.
        electron_free_energy (float): The Gibbs free energy of an electron.
            Defaults to -2.15 eV, the value at which the LiEC SEI forms.
    """

    def __init__(self, input_entries, electron_free_energy=-2.15):

        self.input_entries = input_entries
        self.electron_free_energy = electron_free_energy
        self.entries = {}
        self.entries_list = []

        print(len(self.input_entries),"input entries")

        connected_entries = []
        for entry in self.input_entries:
            # print(len(entry.molecule))
            if len(entry.molecule) > 1:
                if nx.is_weakly_connected(entry.graph):
                    connected_entries.append(entry)
            else:
                connected_entries.append(entry)
        print(len(connected_entries),"connected entries")

        get_formula = lambda x: x.formula
        get_Nbonds = lambda x: x.Nbonds
        get_charge = lambda x: x.charge

        sorted_entries_0 = sorted(connected_entries, key=get_formula)
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
                                    if entry.free_energy() is not None and Uentry.free_energy() is not None:
                                        if entry.free_energy() < Uentry.free_energy():
                                            unique[ii] = entry
                                            # if entry.energy > Uentry.energy:
                                            #     print("WARNING: Free energy lower but electronic energy higher!")
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

        print(len(self.entries_list), "unique entries")

        for ii, entry in enumerate(self.entries_list):
            entry.parameters["ind"] = ii

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(len(self.entries_list)),bipartite=0)

        self.one_electron_redox()
        self.intramol_single_bond_change()
        self.intermol_single_bond_change()
        self.coordination_bond_change()

        self.PR_record = self.build_PR_record()
        self.min_cost = {}

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
                                    # graph0 = frags[0].graph
                                    formula0 = frags[0].molecule.composition.alphabetical_formula
                                    Nbonds0 = len(frags[0].graph.edges())
                                    # graph1 = frags[1].graph
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

    def coordination_bond_change(self):
        # Simultaneous formation / breakage of multiple coordination bonds
        # A + M <-> AM aka AM <-> A + M
        # Three entries with:
        #     M = Li or Mg
        #     comp(AM) = comp(A) + comp(M)
        #     charge(AM) = charge(A) + charge(M)
        #     removing two M-containing edges in AM yields two disconnected subgraphs that are isomorphic to B and C
        M_entries = {}
        for formula in self.entries:
            if formula == "Li1" or formula == "Mg1":
                if formula not in M_entries:
                    M_entries[formula] = {}
                for charge in self.entries[formula][0]:
                    assert(len(self.entries[formula][0][charge])==1)
                    M_entries[formula][charge] = self.entries[formula][0][charge][0]
        if M_entries != {}:
            for formula in self.entries:
                if "Li" in formula or "Mg" in formula:
                    for Nbonds in self.entries[formula]:
                        if Nbonds > 2:
                            for charge in self.entries[formula][Nbonds]:
                                for entry in self.entries[formula][Nbonds][charge]:
                                    nosplit_M_bonds = []
                                    for edge in entry.edges:
                                        if str(entry.molecule.sites[edge[0]].species) in M_entries or str(entry.molecule.sites[edge[1]].species) in M_entries:
                                            M_bond = (edge[0],edge[1])
                                            try:
                                                frags = entry.mol_graph.split_molecule_subgraphs([M_bond], allow_reverse=True)
                                            except MolGraphSplitError:
                                                nosplit_M_bonds.append(M_bond)
                                    bond_pairs = itertools.combinations(nosplit_M_bonds, 2)
                                    for bond_pair in bond_pairs:
                                        try:
                                            frags = entry.mol_graph.split_molecule_subgraphs(bond_pair, allow_reverse=True)
                                            M_ind = None
                                            M_formula = None
                                            for ii,frag in enumerate(frags):
                                                frag_formula = frag.molecule.composition.alphabetical_formula
                                                if frag_formula in M_entries:
                                                    M_ind = ii
                                                    M_formula = frag_formula
                                                    break
                                            if M_ind != None:
                                                for ii, frag in enumerate(frags):
                                                    if ii != M_ind:
                                                        # nonM_graph = frag.graph
                                                        nonM_formula = frag.molecule.composition.alphabetical_formula
                                                        nonM_Nbonds = len(frag.graph.edges())
                                                        if nonM_formula in self.entries:
                                                            if nonM_Nbonds in self.entries[nonM_formula]:
                                                                for nonM_charge in self.entries[nonM_formula][nonM_Nbonds]:
                                                                    M_charge = entry.charge - nonM_charge
                                                                    if M_charge in M_entries[M_formula]:
                                                                        for nonM_entry in self.entries[nonM_formula][nonM_Nbonds][nonM_charge]:
                                                                            if frag.isomorphic_to(nonM_entry):
                                                                                self.add_reaction([entry],[nonM_entry,M_entries[M_formula][M_charge]],"coordination_bond_change")
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
        elif rxn_type == "coordination_bond_change":
            if len(entries0) != 1 or len(entries1) != 2:
                raise RuntimeError("Coordination bond change requires two lists that contain one entry and two entries, respectively!")
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
            if entry1.free_energy() is not None and entry0.free_energy() is not None:
                free_energy_A = entry1.free_energy() - entry0.free_energy()
                free_energy_B = entry0.free_energy() - entry1.free_energy()
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

        elif rxn_type == "intermol_single_bond_change" or rxn_type == "coordination_bond_change":
            entry = entries0[0]
            entry0 = entries1[0]
            entry1 = entries1[1]
            if entry0.parameters["ind"] <= entry1.parameters["ind"]:
                two_mol_name = str(entry0.parameters["ind"])+"+"+str(entry1.parameters["ind"])
            else:
                two_mol_name = str(entry1.parameters["ind"])+"+"+str(entry0.parameters["ind"])
            two_mol_name0 = str(entry0.parameters["ind"])+"+PR_"+str(entry1.parameters["ind"])
            two_mol_name1 = str(entry1.parameters["ind"])+"+PR_"+str(entry0.parameters["ind"])
            node_name_A = str(entry.parameters["ind"])+","+two_mol_name
            node_name_B0 = two_mol_name0+","+str(entry.parameters["ind"])
            node_name_B1 = two_mol_name1+","+str(entry.parameters["ind"])
            if rxn_type == "intermol_single_bond_change":
                rxn_type_A = "Molecular decomposition breaking one bond A -> B+C"
                rxn_type_B = "Molecular formation from one new bond A+B -> C"
            elif rxn_type == "coordination_bond_change":
                rxn_type_A = "Coordination bond breaking AM -> A+M"
                rxn_type_B = "Coordination bond forming A+M -> AM"
            energy_A = entry0.energy + entry1.energy - entry.energy
            energy_B = entry.energy - entry0.energy - entry1.energy
            if entry1.free_energy() is not None and entry0.free_energy() is not None and entry.free_energy() is not None:
                free_energy_A = entry0.free_energy() + entry1.free_energy() - entry.free_energy()
                free_energy_B = entry.free_energy() - entry0.free_energy() - entry1.free_energy()

            self.graph.add_node(node_name_A,rxn_type=rxn_type_A,bipartite=1,energy=energy_A,free_energy=free_energy_A)
            
            self.graph.add_edge(entry.parameters["ind"],
                                node_name_A,
                                softplus=self.softplus(free_energy_A),
                                exponent=self.exponent(free_energy_A),
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

            self.graph.add_node(node_name_B0,rxn_type=rxn_type_B,bipartite=1,energy=energy_B,free_energy=free_energy_B)
            self.graph.add_node(node_name_B1,rxn_type=rxn_type_B,bipartite=1,energy=energy_B,free_energy=free_energy_B)

            self.graph.add_edge(node_name_B0,
                                entry.parameters["ind"],
                                softplus=self.softplus(free_energy_B),
                                exponent=self.exponent(free_energy_B),
                                weight=1.0
                                )
            self.graph.add_edge(node_name_B1,
                                entry.parameters["ind"],
                                softplus=self.softplus(free_energy_B),
                                exponent=self.exponent(free_energy_B),
                                weight=1.0
                                )

            self.graph.add_edge(entry0.parameters["ind"],
                                node_name_B0,
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0
                                )
            self.graph.add_edge(entry1.parameters["ind"],
                                node_name_B1,
                                softplus=0.0,
                                exponent=0.0,
                                weight=1.0)

    def softplus(self, free_energy):
        return np.log(1 + (273.0 / 500.0) * np.exp(free_energy))

    def exponent(self, free_energy):
        return np.exp(free_energy)

    def build_PR_record(self):
        PR_record = {}
        for node in self.graph.nodes():
            if self.graph.node[node]["bipartite"] == 0:
                PR_record[node] = []
        for node in self.graph.nodes():
            if self.graph.node[node]["bipartite"] == 1:
                if "+PR_" in node.split(",")[0]:
                    PR = int(node.split(",")[0].split("+PR_")[1])
                    PR_record[PR].append(node)
        return PR_record

    def characterize_path(self,path,weight,PR_paths={},final=False):
        path_dict = {}
        path_dict["byproducts"] = []
        path_dict["unsolved_prereqs"] = []
        path_dict["solved_prereqs"] = []
        path_dict["all_prereqs"] = []
        path_dict["cost"] = 0.0
        path_dict["path"] = path

        for ii,step in enumerate(path):
            if ii != len(path)-1:
                path_dict["cost"] += self.graph[step][path[ii+1]][weight]
                if ii%2 == 1:
                    rxn = step.split(",")
                    if "+PR_" in rxn[0]:
                        PR = int(rxn[0].split("+PR_")[1])
                        path_dict["all_prereqs"].append(PR)
                    elif "+" in rxn[1]:
                        prods = rxn[1].split("+")
                        if prods[0] == prods[1]:
                            path_dict["byproducts"].append(int(prods[0]))
                        else:
                            for prod in prods:
                                if int(prod) != path[ii+1]:
                                    path_dict["byproducts"].append(int(prod))
        for PR in path_dict["all_prereqs"]:
            if PR in path_dict["byproducts"]:
                # Note that we're ignoring the order in which BPs are made vs they come up as PRs...
                path_dict["all_prereqs"].remove(PR)
                path_dict["byproducts"].remove(PR)
                if PR in self.min_cost:
                    path_dict["cost"] -= self.min_cost[PR]
        for PR in path_dict["all_prereqs"]:
            if PR in PR_paths:
                path_dict["solved_prereqs"].append(PR)
            else:
                path_dict["unsolved_prereqs"].append(PR)

        if final:
            path_dict["overall_free_energy_change"] = 0.0
            path_dict["hardest_step"] = None
            path_dict["description"] = ""

            assert(len(path_dict["solved_prereqs"])==len(path_dict["all_prereqs"]))
            assert(len(path_dict["unsolved_prereqs"])==0)
            del path_dict["solved_prereqs"]
            del path_dict["unsolved_prereqs"]

            PRs_to_join = copy.deepcopy(path_dict["all_prereqs"])
            full_path = copy.deepcopy(path)
            while len(PRs_to_join) > 0:
                new_PRs = []
                for PR in PRs_to_join:
                    PR_path = PR_paths[PR]
                    assert(len(PR_path["solved_prereqs"])==len(PR_path["all_prereqs"]))
                    for new_PR in PR_path["all_prereqs"]:
                        new_PRs.append(new_PR)
                        path_dict["all_prereqs"].append(new_PR)
                    for new_BP in PR_path["byproducts"]:
                        path_dict["byproducts"].append(new_BP)
                    full_path = PR_path["path"] + full_path
                PRs_to_join = copy.deepcopy(new_PRs)

            for PR in path_dict["all_prereqs"]:
                if PR in path_dict["byproducts"]:
                    print("WARNING: Matching prereq and byproduct found!")

            for ii,step in enumerate(full_path):
                if self.graph.node[step]["bipartite"] == 1:
                    path_dict["overall_free_energy_change"] += self.graph.node[step]["free_energy"]
                    if path_dict["description"] == "":
                        path_dict["description"] += self.graph.node[step]["rxn_type"]
                    else:
                        path_dict["description"] += ", " + self.graph.node[step]["rxn_type"]
                    if path_dict["hardest_step"] == None:
                        path_dict["hardest_step"] = step
                    elif self.graph.node[step]["free_energy"] > self.graph.node[path_dict["hardest_step"]]["free_energy"]:
                        path_dict["hardest_step"] = step
            del path_dict["path"]
            path_dict["full_path"] = full_path
            path_dict["hardest_step_deltaG"] = self.graph.node[path_dict["hardest_step"]]["free_energy"]
        return path_dict

    def solve_prerequisites(self,starts,target,weight):
        PRs = {}
        old_solved_PRs = []
        new_solved_PRs = ["placeholder"]
        no_path_to = {}
        orig_graph = copy.deepcopy(self.graph)
        ii = 0
        for start in starts:
            no_path_to[start] = []
        for start in starts:
            PRs[start] = self.characterize_path([start],weight)
        print(PRs)
        while len(new_solved_PRs) > 0:
            min_cost = {}
            for PR in PRs:
                min_cost[PR] = PRs[PR]["cost"]
            for start in starts:
                for node in self.graph.nodes():
                    if self.graph.node[node]["bipartite"] == 0 and node not in old_solved_PRs and node != target and node not in no_path_to[start]:
                        path_exists = True
                        try:
                            length,dij_path = nx.algorithms.simple_paths._bidirectional_dijkstra(
                                self.graph,
                                source=hash(start),
                                target=hash(node),
                                ignore_nodes=self.find_or_remove_bad_nodes([target,node]),
                                weight=weight)
                        except nx.exception.NetworkXNoPath:
                            # print("No path exists to node ",node)
                            no_path_to[start].append(node)
                            path_exists = False
                        if path_exists:
                            if len(dij_path) > 1 and len(dij_path)%2 == 1:
                                path = self.characterize_path(dij_path,weight,PRs)
                                if len(path["unsolved_prereqs"]) == 0:
                                    if node in PRs:
                                        if path["cost"] < PRs[node]["cost"]:
                                            PRs[node] = path
                                            min_cost[node] = path["cost"]
                                    else:
                                        PRs[node] = path
                                        min_cost[node] = path["cost"]
                                elif node in min_cost:
                                    if path["cost"] < min_cost[node]:
                                        min_cost[node] = path["cost"]
                                else:
                                    min_cost[node] = path["cost"]
                                    # print(node, path["prereqs"], len(path["path"]), path["cost"])

            solved_PRs = list(PRs.keys())
            new_solved_PRs = []
            for PR in solved_PRs:
                if PR not in old_solved_PRs:
                    new_solved_PRs.append(PR)
            # print()
            # print(ii,old_solved_PRs,new_solved_PRs)
            print(ii,len(old_solved_PRs),len(new_solved_PRs))
            attrs = {}

            for PR_ind in min_cost:
                for node in self.PR_record[PR_ind]:
                    split_node = node.split(",")
                    attrs[(node,int(split_node[1]))] = {weight:orig_graph[node][int(split_node[1])][weight]+min_cost[PR_ind]}
            nx.set_edge_attributes(self.graph,attrs)
            self.min_cost = copy.deepcopy(min_cost)
            old_solved_PRs = copy.deepcopy(solved_PRs)
            ii += 1
        return PRs

    def find_or_remove_bad_nodes(self,nodes,remove_nodes=False):
        bad_nodes = []
        for node in nodes:
            for bad_node in self.PR_record[node]:
                bad_nodes.append(bad_node)
        if remove_nodes:
            pruned_graph = copy.deepcopy(self.graph)
            pruned_graph.remove_nodes_from(bad_nodes)
            return pruned_graph
        else:
            return bad_nodes

    def valid_shortest_simple_paths(self,start,target,weight,PRs=[]):
        bad_nodes = PRs
        bad_nodes.append(target)
        valid_graph = self.find_or_remove_bad_nodes(bad_nodes,remove_nodes=True)
        return nx.shortest_simple_paths(valid_graph,hash(start),hash(target),weight=weight)

    def find_paths(self,starts,target,weight,num_paths=10):
        """
        Args:
            starts ([int]): List of starting node IDs (ints). 
            target (int): Target node ID.
            weight (str): String identifying what edge weight to use for path finding.
            num_paths (int): Number of paths to find. Defaults to 10.
        """
        paths = []
        c = itertools.count()
        my_heapq = []

        print("Solving prerequisites...")
        PR_paths = self.solve_prerequisites(starts,target,weight)

        print("Finding paths...")
        for start in starts:
            ind = 0
            for path in self.valid_shortest_simple_paths(start,target,weight):
                if ind == num_paths:
                    break
                else:
                    ind += 1
                    path_dict = self.characterize_path(path,weight,PR_paths,final=True)
                    heapq.heappush(my_heapq, (path_dict["cost"],next(c),path_dict))

        while len(paths) < num_paths and my_heapq:
            (cost, _, path_dict) = heapq.heappop(my_heapq)
            print(len(paths),cost,len(my_heapq),path_dict["all_prereqs"])
            paths.append(path_dict)

        return PR_paths, paths

    def identify_sinks(self):
        sinks = []
        for node in self.graph.nodes():
            if self.graph.node[node]["bipartite"] == 0:
                neighbor_list = list(self.graph.neighbors(node))
                if len(neighbor_list) > 0:
                    neg_found = False
                    for neighbor in neighbor_list:
                        if self.graph.node[neighbor]["free_energy"] < 0:
                            neg_found = True
                            break
                    if not neg_found:
                        sinks.append(node)
        return sinks

