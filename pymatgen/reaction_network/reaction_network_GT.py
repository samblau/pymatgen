import json
import time
from json import JSONEncoder

from networkx.readwrite import json_graph
#import time
import yaml
from networkx.readwrite import json_graph
import networkx.algorithms.isomorphism as iso

from abc import ABCMeta, abstractproperty, abstractmethod, abstractclassmethod
from abc import ABC, abstractmethod
#from time import time
from gunicorn.util import load_class
from monty.json import MSONable
import logging
import copy
import itertools
import heapq
import numpy as np
from monty.json import MSONable, MontyDecoder
from monty.serialization import dumpfn, loadfn

from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen import Molecule
from pymatgen.analysis.fragmenter import metal_edge_extender
import networkx as nx
from networkx.algorithms import bipartite
from pymatgen.entries.mol_entry import MoleculeEntry
from pymatgen.core.composition import CompositionError
from typing import List, Dict, Tuple, Generator
from graph_tool.all import *

MappingDict = Dict[str, Dict[int, Dict[int, List[MoleculeEntry]]]]
Mapping_Energy_Dict = Dict[str, float]
Mapping_ReactionType_Dict = Dict[str, str]
Mapping_Record_Dict = Dict[str, List[str]]


class Reaction(MSONable, metaclass=ABCMeta):
    """
       Abstract class for subsequent types of reaction class

       Args:
           reactants ([MoleculeEntry]): A list of MoleculeEntry objects of len 1.
           products ([MoleculeEntry]): A list of MoleculeEntry objects of max len 2
       """

    def __init__(self, reactants: List[MoleculeEntry], products: List[MoleculeEntry]):
        self.reactants = reactants
        self.products = products
        self.entry_ids = {e.entry_id for e in self.reactants}

    def __in__(self, entry: MoleculeEntry):
        return entry.entry_id in self.entry_ids

    def __len__(self):
        return len(self.reactants)

    @classmethod
    @abstractmethod
    def generate(cls, entries: MappingDict):
        pass

    @abstractmethod
    def graph_representation(self, input_graph) -> nx.DiGraph:
        pass

    @abstractmethod
    def reaction_type(self) -> Mapping_ReactionType_Dict:
        pass

    @abstractmethod
    def energy(self) -> Mapping_Energy_Dict:
        pass

    @abstractmethod
    def free_energy(self) -> Mapping_Energy_Dict:
        pass

    @abstractmethod
    def rate(self):
        pass


#     @abstractmethod
#     def num_entries(self):
#         """
# 		number of molecule entries that are intertacting
# 		num reactants + num products not including electron
# 		"""
#
#     @abstractmethod
#     def virtual_edges(self, from_entry):
#         """
# 		Returns the virtual networkx edge that goes from the molecule node to the PR reaction node
# 		Virtual edge - (from_node,to_node)
# 		"""
#         pass
#
#     @abstractmethod
#     def update_weights(self, preq_costs: Dict[str, float]) -> Dict[Tuple(str, str), Dict[str, float]]:
#         """
#
# 		Arguments:
# 			prereq_costs: dictionary mapping entry_id to the new/delta cost
#
# 		Returns:
# 			Dictionary to update weights of edges in networkx graph
# 			Dictionary key is a tuple for the two nodes that define the edge
# 			Dictionary value is a dictionary with key of the weight function(s)
# 				and value of the new weight in new/delta cost
# # 		"""
class GraphTool(MSONable):
    def __init__(self):
        self.graph = Graph(directed=True)
        self.node_label = self.graph.new_vertex_property("string")
        self.bipartite = self.graph.new_vertex_property("int32_t")
        self.energy = self.graph.new_vertex_property("double")
        self.entry_ids = self.graph.new_vertex_property("string")
        self.free_energy = self.graph.new_vertex_property("double")
        self.rxn_type = self.graph.new_vertex_property("string")
        self.exponent = self.graph.new_edge_property("double")
        self.softplus = self.graph.new_edge_property("double")
        self.weight = self.graph.new_edge_property("double")
        self.vfilter = self.graph.new_vertex_property("boolean")
        self.efilter = self.graph.new_edge_property("boolean")
        self.vind_to_vlabel_dict = {int(n): self.graph.vp["node_label"][n] for n in self.graph.vertices()}
        self.vlabel_to_vind_dict = {self.graph.vp["node_label"][n]:int(n) for n in self.graph.vertices()}

    def from_gt_to_GraphTool(self, gt_graph):
        class_instance = GraphTool()
        class_instance.graph = copy.deepcopy(gt_graph)
        print("!!",class_instance.graph.num_vertices(ignore_filter=True))
        class_instance.node_label = class_instance.graph.vertex_properties["node_label"]
        class_instance.bipartite = class_instance.graph.vertex_properties["bipartite"]
        class_instance.energy = class_instance.graph.vertex_properties["energy"]
        class_instance.entry_ids = class_instance.graph.vertex_properties["entry_ids"]
        class_instance.free_energy = class_instance.graph.vertex_properties["free_energy"]
        class_instance.rxn_type = class_instance.graph.vertex_properties["rxn_type"]
        class_instance.exponent = class_instance.graph.edge_properties["exponent"]
        class_instance.softplus = class_instance.graph.edge_properties["softplus"]
        class_instance.weight = class_instance.graph.edge_properties["weight"]
        class_instance.vfilter = class_instance.graph.vertex_properties["vfliter"]
        class_instance.efilter = class_instance.graph.edge_properties["efliter"]
        class_instance.vind_to_vlabel_dict = {int(n): class_instance.graph.vp["node_label"][n] for n in class_instance.graph.vertices()}
        class_instance.vlabel_to_vind_dict = {class_instance.graph.vp["node_label"][n]:int(n) for n in class_instance.graph.vertices()}

        return class_instance
    @property
    def as_dict(self) -> dict:
        """
            A method to convert ReactionPath objection into a dictionary
        :return: d: dictionary containing all te ReactionPath attributes
        """
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "graph": self.graph,
             "node_label": self.node_label,
             "bipartite": self.bipartite,
             "energy": self.energy,
             "entry_ids": self.entry_ids,
             "free_energy": self.free_energy,
             "rxn_type": self.rxn_type,
             "exponent": self.exponent,
             "softplus": self.softplus,
             "weight": self.weight,
             "vfilter": self.vfilter,
             "efilter": self.efilter,
             "vind_to_vlabel_dict": self.vind_to_vlabel_dict,
             "vlabel_to_vind_dict":self.vlabel_to_vind_dict
             }
        return d

    @classmethod
    def from_dict(cls, d):
        """
            A method to convert dict to ReactionPath object
        :param d:  dict retuend from ReactionPath.as_dict() method
        :return: ReactionPath object
        """
        x = cls()
        x.graph = d.get("graph")
        x.node_label = d.get("node_label")
        x.bipartite = d.get("bipartite")
        x.energy = d.get("energy")
        x.entry_ids = d.get("entry_ids")
        x.free_energy = d.get("free_energy")
        x.rxn_type = d.get("rxn_type")
        x.exponent = d.get("exponent")
        x.softplus = d.get("softplus")
        x.weight = d.get("weight")
        x.vfilter = d.get("vfilter")
        x.efilter = d.get("efilter")
        x.vind_to_vlabel_dict = d.get("vind_to_vlabel_dict")
        x.vlabel_to_vind_dict = d.get("vlabel_to_vind_dict")
        return x

    def ignorenodes(self, node_list):
        v_toremove = []

        for n in node_list:
            find_v = graph_tool.util.find_vertex(self.graph, self.node_label, n)
            v_toremove.append(int(find_v[0]))
        for v in self.graph.vertices():
            self.vfilter[v] = True
        for v in v_toremove:
            self.vfilter[v] = False
        gv = GraphView(self.graph, vfilt=self.vfilter)
        return gv

# def graph_rep_2_2(reaction: Reaction) -> nx.DiGraph:
#     """
#     A method to convert a reaction type object into graph representation. Reaction much be of type 2 reactants -> 2
#     products
#     Args:
#        :param reaction: (any of the reaction class object, ex. RedoxReaction, IntramolSingleBondChangeReaction, Concerted)
#     """
#
#     if len(reaction.reactants) != 2 or len(reaction.products) != 2:
#         raise ValueError("Must provide reaction with 2 reactants and 2 products for graph_rep_2_2")
#
#     reactant_0 = reaction.reactants[0]
#     reactant_1 = reaction.reactants[1]
#     product_0 = reaction.products[0]
#     product_1 = reaction.products[1]
#     graph = nx.DiGraph()
#
#     if product_0.parameters["ind"] <= product_1.parameters["ind"]:
#         two_prod_name = str(product_0.parameters["ind"]) + "+" + str(product_1.parameters["ind"])
#         two_prod_name_entry_ids = str(product_0.entry_id) + "+" + str(product_1.entry_id)
#     else:
#         two_prod_name = str(product_1.parameters["ind"]) + "+" + str(product_0.parameters["ind"])
#         two_prod_name_entry_ids = str(product_1.entry_id) + "+" + str(product_0.entry_id)
#
#     if reactant_0.parameters["ind"] <= reactant_1.parameters["ind"]:
#         two_reac_name = str(reactant_0.parameters["ind"]) + "+" + str(reactant_1.parameters["ind"])
#         two_reac_name_entry_ids = str(reactant_0.entry_id) + "+" + str(reactant_1.entry_id)
#     else:
#         two_reac_name = str(reactant_1.parameters["ind"]) + "+" + str(reactant_0.parameters["ind"])
#         two_reac_name_entry_ids = str(reactant_1.entry_id) + "+" + str(reactant_0.entry_id)
#
#     two_prod_name0 = str(product_0.parameters["ind"]) + "+PR_" + str(product_1.parameters["ind"])
#     two_prod_name1 = str(product_1.parameters["ind"]) + "+PR_" + str(product_0.parameters["ind"])
#
#     two_reac_name0 = str(reactant_0.parameters["ind"]) + "+PR_" + str(reactant_1.parameters["ind"])
#     two_reac_name1 = str(reactant_1.parameters["ind"]) + "+PR_" + str(reactant_0.parameters["ind"])
#
#     node_name_A0 = two_reac_name0 + "," + two_prod_name
#     node_name_A1 = two_reac_name1 + "," + two_prod_name
#     node_name_B0 = two_prod_name0 + "," + two_reac_name
#     node_name_B1 = two_prod_name1 + "," + two_reac_name
#
#     two_prod_entry_ids0 = str(product_0.entry_id) + "+PR_" + str(product_1.entry_id)
#     two_prod_entry_ids1 = str(product_1.entry_id) + "+PR_" + str(product_0.entry_id)
#
#     two_reac_entry_ids0 = str(reactant_0.entry_id) + "+PR_" + str(reactant_1.entry_id)
#     two_reac_entry_ids1 = str(reactant_1.entry_id) + "+PR_" + str(reactant_0.entry_id)
#
#     entry_ids_name_A0 = two_reac_entry_ids0 + "," + two_prod_name_entry_ids
#     entry_ids_name_A1 = two_reac_entry_ids1 + "," + two_prod_name_entry_ids
#     entry_ids_name_B0 = two_prod_entry_ids0 + "," + two_reac_name_entry_ids
#     entry_ids_name_B1 = two_prod_entry_ids1 + "," + two_reac_name_entry_ids
#
#     rxn_type_A = reaction.reaction_type()["rxn_type_A"]
#     rxn_type_B = reaction.reaction_type()["rxn_type_B"]
#     energy_A = reaction.energy()["energy_A"]
#     energy_B = reaction.energy()["energy_B"]
#     free_energy_A = reaction.free_energy()["free_energy_A"]
#     free_energy_B = reaction.free_energy()["free_energy_B"]
#
#     graph.add_node(node_name_A0, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
#                    entry_ids=entry_ids_name_A0)
#
#     graph.add_edge(reactant_0.parameters["ind"],
#                    node_name_A0,
#                    softplus=ReactionNetwork.softplus(free_energy_A),
#                    exponent=ReactionNetwork.exponent(free_energy_A),
#                    weight=1.0
#                    )
#
#     graph.add_edge(node_name_A0,
#                    product_0.parameters["ind"],
#                    softplus=0.0,
#                    exponent=0.0,
#                    weight=1.0
#                    )
#     graph.add_edge(node_name_A0,
#                    product_1.parameters["ind"],
#                    softplus=0.0,
#                    exponent=0.0,
#                    weight=1.0
#                    )
#
#     graph.add_node(node_name_A1, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
#                    entry_ids=entry_ids_name_A1)
#
#     graph.add_edge(reactant_1.parameters["ind"],
#                    node_name_A1,
#                    softplus=ReactionNetwork.softplus(free_energy_A),
#                    exponent=ReactionNetwork.exponent(free_energy_A),
#                    weight=1.0
#                    )
#
#     graph.add_edge(node_name_A1,
#                    product_0.parameters["ind"],
#                    softplus=0.0,
#                    exponent=0.0,
#                    weight=1.0
#                    )
#     graph.add_edge(node_name_A1,
#                    product_1.parameters["ind"],
#                    softplus=0.0,
#                    exponent=0.0,
#                    weight=1.0
#                    )
#
#     graph.add_node(node_name_B0, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
#                    entry_ids=entry_ids_name_B0)
#
#     graph.add_edge(product_0.parameters["ind"],
#                    node_name_B0,
#                    softplus=ReactionNetwork.softplus(free_energy_B),
#                    exponent=ReactionNetwork.exponent(free_energy_B),
#                    weight=1.0
#                    )
#
#     graph.add_edge(node_name_B0,
#                    reactant_0.parameters["ind"],
#                    softplus=0.0,
#                    exponent=0.0,
#                    weight=1.0
#                    )
#     graph.add_edge(node_name_B0,
#                    reactant_1.parameters["ind"],
#                    softplus=0.0,
#                    exponent=0.0,
#                    weight=1.0
#                    )
#
#     graph.add_node(node_name_B1, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
#                    entry_ids=entry_ids_name_B1)
#
#     graph.add_edge(product_1.parameters["ind"],
#                    node_name_B1,
#                    softplus=ReactionNetwork.softplus(free_energy_B),
#                    exponent=ReactionNetwork.exponent(free_energy_B),
#                    weight=1.0
#                    )
#
#     graph.add_edge(node_name_B1,
#                    reactant_0.parameters["ind"],
#                    softplus=0.0,
#                    exponent=0.0,
#                    weight=1.0
#                    )
#     graph.add_edge(node_name_B1,
#                    reactant_1.parameters["ind"],
#                    softplus=0.0,
#                    exponent=0.0,
#                    weight=1.0
#                    )
#
#     return graph


def graph_rep_1_2(reaction: Reaction, input_graph) -> nx.DiGraph:
    """
    A method to convert a reaction type object into graph representation. Reaction much be of type 1 reactant -> 2
    products

    Args:
       :param reaction: (any of the reaction class object, ex. RedoxReaction, IntramolSingleBondChangeReaction)
    """

    if len(reaction.reactants) != 1 or len(reaction.products) != 2:
        raise ValueError("Must provide reaction with 1 reactant and 2 products for graph_rep_1_2")

    reactant_0 = reaction.reactants[0]
    product_0 = reaction.products[0]
    product_1 = reaction.products[1]
    #graph = nx.DiGraph()

    if product_0.parameters["ind"] <= product_1.parameters["ind"]:
        two_mol_name = str(product_0.parameters["ind"]) + "+" + str(product_1.parameters["ind"])
        two_mol_name_entry_ids = str(product_0.entry_id) + "+" + str(product_1.entry_id)
    else:
        two_mol_name = str(product_1.parameters["ind"]) + "+" + str(product_0.parameters["ind"])
        two_mol_name_entry_ids = str(product_1.entry_id) + "+" + str(product_0.entry_id)

    two_mol_name0 = str(product_0.parameters["ind"]) + "+PR_" + str(product_1.parameters["ind"])
    two_mol_name1 = str(product_1.parameters["ind"]) + "+PR_" + str(product_0.parameters["ind"])
    node_name_A = str(reactant_0.parameters["ind"]) + "," + two_mol_name
    node_name_B0 = two_mol_name0 + "," + str(reactant_0.parameters["ind"])
    node_name_B1 = two_mol_name1 + "," + str(reactant_0.parameters["ind"])

    two_mol_entry_ids0 = str(product_0.entry_id) + "+PR_" + str(product_1.entry_id)
    two_mol_entry_ids1 = str(product_1.entry_id) + "+PR_" + str(product_0.entry_id)
    entry_ids_name_A = str(reactant_0.entry_id) + "," + two_mol_name_entry_ids
    entry_ids_name_B0 = two_mol_entry_ids0 + "," + str(reactant_0.entry_id)
    entry_ids_name_B1 = two_mol_entry_ids1 + "," + str(reactant_0.entry_id)

    rxn_type_A = reaction.reaction_type()["rxn_type_A"]
    rxn_type_B = reaction.reaction_type()["rxn_type_B"]
    energy_A = reaction.energy()["energy_A"]
    energy_B = reaction.energy()["energy_B"]
    free_energy_A = reaction.free_energy()["free_energy_A"]
    free_energy_B = reaction.free_energy()["free_energy_B"]
    # graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
    #                entry_ids=entry_ids_name_A)
    #
    # graph.add_edge(reactant_0.parameters["ind"],
    #                node_name_A,
    #                softplus=ReactionNetwork.softplus(free_energy_A),
    #                exponent=ReactionNetwork.exponent(free_energy_A),
    #                weight=1.0
    #                )
    #
    # graph.add_edge(node_name_A,
    #                product_0.parameters["ind"],
    #                softplus=0.0,
    #                exponent=0.0,
    #                weight=1.0
    #                )
    # graph.add_edge(node_name_A,
    #                product_1.parameters["ind"],
    #                softplus=0.0,
    #                exponent=0.0,
    #                weight=1.0
    #                )
    #
    # graph.add_node(node_name_B0, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
    #                entry_ids=entry_ids_name_B0)
    # graph.add_node(node_name_B1, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
    #                entry_ids=entry_ids_name_B1)
    #
    # graph.add_edge(node_name_B0,
    #                reactant_0.parameters["ind"],
    #                softplus=0.0,
    #                exponent=0.0,
    #                weight=1.0
    #                )
    # graph.add_edge(node_name_B1,
    #                reactant_0.parameters["ind"],
    #                softplus=0.0,
    #                exponent=0.0,
    #                weight=1.0
    #                )
    #
    # graph.add_edge(product_0.parameters["ind"],
    #                node_name_B0,
    #                softplus=ReactionNetwork.softplus(free_energy_B),
    #                exponent=ReactionNetwork.exponent(free_energy_B),
    #                weight=1.0
    #                )
    # graph.add_edge(product_1.parameters["ind"],
    #                node_name_B1,
    #                softplus=ReactionNetwork.softplus(free_energy_B),
    #                exponent=ReactionNetwork.exponent(free_energy_B),
    #                weight=1.0)


    ########
    if graph_tool.util.find_vertex(input_graph.graph, input_graph.node_label, node_name_A) == []:

        v_a = input_graph.graph.add_vertex()
        input_graph.node_label[v_a] = node_name_A
        input_graph.rxn_type[v_a] = rxn_type_A
        input_graph.bipartite[v_a] = 1
        input_graph.energy[v_a] = energy_A
        input_graph.free_energy[v_a] = free_energy_A
        input_graph.entry_ids[v_a] = entry_ids_name_A
    else:
        v_a = graph_tool.util.find_vertex(input_graph.graph, input_graph.node_label, node_name_A)[0]



    if input_graph.graph.edge(int(reactant_0.parameters["ind"]), int(v_a)) is None:
        eg_a_1 = input_graph.graph.add_edge(int(reactant_0.parameters["ind"]), int(v_a), add_missing=False)
        input_graph.softplus[eg_a_1] = ReactionNetwork.softplus(free_energy_A)
        input_graph.exponent[eg_a_1] = ReactionNetwork.softplus(free_energy_A)
        input_graph.weight[eg_a_1] = 1

    if input_graph.graph.edge(int(v_a), int(product_0.parameters["ind"])) is None:
        eg_a_2 = input_graph.graph.add_edge(int(v_a), int(product_0.parameters["ind"]), add_missing=False)
        input_graph.softplus[eg_a_2] = 0
        input_graph.exponent[eg_a_2] = 0
        input_graph.weight[eg_a_2] = 1
    if input_graph.graph.edge(int(v_a), int(product_1.parameters["ind"])) is None:
        eg_a_3 = input_graph.graph.add_edge(int(v_a), int(product_1.parameters["ind"]), add_missing=False)
        input_graph.softplus[eg_a_3] = 0
        input_graph.exponent[eg_a_3] = 0
        input_graph.weight[eg_a_3] = 1

    if graph_tool.util.find_vertex(input_graph.graph, input_graph.node_label, node_name_B0) == []:
        v_b0 = input_graph.graph.add_vertex()
        input_graph.node_label[v_b0] = node_name_B0
        input_graph.rxn_type[v_b0] = rxn_type_B
        input_graph.bipartite[v_b0] = 1
        input_graph.energy[v_b0] = energy_B
        input_graph.free_energy[v_b0] = free_energy_B
        input_graph.entry_ids[v_b0] = entry_ids_name_B0
    else:
        v_b0 = graph_tool.util.find_vertex(input_graph.graph, input_graph.node_label, node_name_B0)[0]

    if graph_tool.util.find_vertex(input_graph.graph, input_graph.node_label, node_name_B1) == []:
        v_b1 = input_graph.graph.add_vertex()
        input_graph.node_label[v_b1] = node_name_B1
        input_graph.rxn_type[v_b1] = rxn_type_B
        input_graph.bipartite[v_b1] = 1
        input_graph.energy[v_b1] = energy_B
        input_graph.free_energy[v_b1] = free_energy_B
        input_graph.entry_ids[v_b1] = entry_ids_name_B1

    else:
        v_b1 = graph_tool.util.find_vertex(input_graph.graph, input_graph.node_label, node_name_B1)[0]



    if input_graph.graph.edge(int(v_b0),int(reactant_0.parameters["ind"])) is None:
        eg_b_0 = input_graph.graph.add_edge(int(v_b0),int(reactant_0.parameters["ind"]), add_missing=False)
        input_graph.softplus[eg_b_0] = 0
        input_graph.exponent[eg_b_0] = 0
        input_graph.weight[eg_b_0] = 1

    if input_graph.graph.edge(int(v_b1),int(reactant_0.parameters["ind"])) is None:
        eg_b_1 = input_graph.graph.add_edge(int(v_b1),int(reactant_0.parameters["ind"]), add_missing=False)
        input_graph.softplus[eg_b_1] = 0
        input_graph.exponent[eg_b_1] = 0
        input_graph.weight[eg_b_1] = 1

    if input_graph.graph.edge(int(product_0.parameters["ind"]), int(v_b0)) is None:
        eg_b_2 = input_graph.graph.add_edge(int(product_0.parameters["ind"]), int(v_b0), add_missing=False)
        input_graph.softplus[eg_b_2] = ReactionNetwork.softplus(free_energy_B)
        input_graph.exponent[eg_b_2] = ReactionNetwork.softplus(free_energy_B)
        input_graph.weight[eg_b_2] = 1

    if input_graph.graph.edge(int(product_1.parameters["ind"]), int(v_b1)) is None:
        eg_b_3 = input_graph.graph.add_edge(int(product_1.parameters["ind"]), int(v_b1), add_missing=False)
        input_graph.softplus[eg_b_3] = ReactionNetwork.softplus(free_energy_B)
        input_graph.exponent[eg_b_3] = ReactionNetwork.softplus(free_energy_B)
        input_graph.weight[eg_b_3] = 1
    return input_graph


def graph_rep_1_1(reaction: Reaction, input_graph) -> nx.DiGraph:
    """
    A method to convert a reaction type object into graph representation. Reaction much be of type 1 reactant -> 1
    product

    Args:
       :param reaction:(any of the reaction class object, ex. RedoxReaction, IntramolSingleBondChangeReaction)
    """

    if len(reaction.reactants) != 1 or len(reaction.products) != 1:
        raise ValueError("Must provide reaction with 1 reactant and product for graph_rep_1_1")

    reactant_0 = reaction.reactants[0]
    product_0 = reaction.products[0]
    graph = nx.DiGraph()
    node_name_A = str(reactant_0.parameters["ind"]) + "," + str(product_0.parameters["ind"])
    node_name_B = str(product_0.parameters["ind"]) + "," + str(reactant_0.parameters["ind"])
    rxn_type_A = reaction.reaction_type()["rxn_type_A"]
    rxn_type_B = reaction.reaction_type()["rxn_type_B"]
    energy_A = reaction.energy()["energy_A"]
    energy_B = reaction.energy()["energy_B"]
    free_energy_A = reaction.free_energy()["free_energy_A"]
    free_energy_B = reaction.free_energy()["free_energy_B"]
    entry_ids_A = str(reactant_0.entry_id) + "," + str(product_0.entry_id)
    entry_ids_B = str(product_0.entry_id) + "," + str(reactant_0.entry_id)

    # graph.add_node(node_name_A, rxn_type=rxn_type_A, bipartite=1, energy=energy_A, free_energy=free_energy_A,
    #                entry_ids=entry_ids_A)
    # graph.add_edge(reactant_0.parameters["ind"],
    #                node_name_A,
    #                softplus=ReactionNetwork.softplus(free_energy_A),
    #                exponent=ReactionNetwork.exponent(free_energy_A),
    #                weight=1.0)
    # graph.add_edge(node_name_A,
    #                product_0.parameters["ind"],
    #                softplus=0.0,
    #                exponent=0.0,
    #                weight=1.0)
    # graph.add_node(node_name_B, rxn_type=rxn_type_B, bipartite=1, energy=energy_B, free_energy=free_energy_B,
    #                entry_ids=entry_ids_B)
    # graph.add_edge(product_0.parameters["ind"],
    #                node_name_B,
    #                softplus=ReactionNetwork.softplus(free_energy_B),
    #                exponent=ReactionNetwork.exponent(free_energy_B),
    #                weight=1.0)
    # graph.add_edge(node_name_B,
    #                reactant_0.parameters["ind"],
    #                softplus=0.0,
    #                exponent=0.0,
    #                weight=1.0)
    #### graph tool stuff
    #gt_graph = GraphTool()
    #gt_graph = copy.deepcopy(input_graph)
    if graph_tool.util.find_vertex(input_graph.graph, input_graph.node_label, node_name_A) == []:
        v_a = input_graph.graph.add_vertex()
        input_graph.node_label[v_a] = node_name_A
        input_graph.rxn_type[v_a] = rxn_type_A
        input_graph.bipartite[v_a] = 1
        input_graph.energy[v_a] = energy_A
        input_graph.free_energy[v_a] = free_energy_A
        input_graph.entry_ids[v_a] = entry_ids_A
    else:
        v_a = graph_tool.util.find_vertex(input_graph.graph, input_graph.node_label, node_name_A)[0]

    # input_graph.vind_to_vlabel_dict = {int(n): input_graph.node_label[n] for n in input_graph.graph.vertices()}
    # input_graph.vlabel_to_vind_dict = {input_graph.node_label[n]: int(n) for n in input_graph.graph.vertices()}
    # print("@#", input_graph.vind_to_vlabel_dict)
    # print("$%", input_graph.vlabel_to_vind_dict)
    if input_graph.graph.edge(int(reactant_0.parameters["ind"]), int(v_a)) is None:
        eg_a_1 = input_graph.graph.add_edge(int(reactant_0.parameters["ind"]), int(v_a), add_missing=False)
        input_graph.softplus[eg_a_1] = ReactionNetwork.softplus(free_energy_A)
        input_graph.exponent[eg_a_1] = ReactionNetwork.softplus(free_energy_A)
        input_graph.weight[eg_a_1] = 1
    if input_graph.graph.edge(int(v_a),int(product_0.parameters["ind"])) is None:
        eg_a_2 = input_graph.graph.add_edge(int(v_a),int(product_0.parameters["ind"]), add_missing=False)
        input_graph.softplus[eg_a_2] = 0
        input_graph.exponent[eg_a_2] = 0
        input_graph.weight[eg_a_2] = 1

    if graph_tool.util.find_vertex(input_graph.graph, input_graph.node_label, node_name_B) == []:
        v_b = input_graph.graph.add_vertex()
        input_graph.node_label[v_b] = node_name_B
        input_graph.rxn_type[v_b] = rxn_type_B
        input_graph.bipartite[v_b] = 1
        input_graph.energy[v_b] = energy_B
        input_graph.free_energy[v_b] = free_energy_B
        input_graph.entry_ids[v_b] = entry_ids_B
    else:
        v_b = graph_tool.util.find_vertex(input_graph.graph, input_graph.node_label, node_name_B)[0]



    if input_graph.graph.edge(int(product_0.parameters["ind"]), int(v_b)) is None:
        eg_b_1 = input_graph.graph.add_edge(int(product_0.parameters["ind"]), int(v_b), add_missing=False)
        input_graph.softplus[eg_b_1] = ReactionNetwork.softplus(free_energy_B)
        input_graph.exponent[eg_b_1] = ReactionNetwork.softplus(free_energy_B)
        input_graph.weight[eg_b_1] = 1
    if input_graph.graph.edge(int(v_b),int(reactant_0.parameters["ind"])) is None:
        eg_b_2 = input_graph.graph.add_edge(int(v_b),int(reactant_0.parameters["ind"]), add_missing=False)
        input_graph.softplus[eg_b_2] = 0
        input_graph.exponent[eg_b_2] = 0
        input_graph.weight[eg_b_2] = 1


    #print(graph_tool.util.find_vertex(input_graph.graph, input_graph.node_label, str(reactant_0.parameters["ind"])))
    # print(list(input_graph.graph.vertices()))
    # print(list(input_graph.graph.edges()))
    # for n in gt_graph.graph.vertices():
    #     print(gt_graph.graph.vp["bipartite"][n])
    #bipartite_dict = {n: gt_graph.graph.vp["bipartite"][n] for n in gt_graph.graph.vertices()}
    #print(bipartite_dict)
    # print("@@")
    #print()
    return input_graph


class RedoxReaction(Reaction):
    """
    A class to define redox reactions as follows:
    One electron oxidation / reduction without change to bonding
        A^n ±e- <-> A^n±1
        Two entries with:
        identical composition
        identical number of edges
        a charge difference of 1
        isomorphic molecule graphs

    Args:
       reactant([MolecularEntry]): list of single molecular entry
       product([MoleculeEntry]): list of single molecular entry
    """

    def __init__(self, reactant: MoleculeEntry, product: MoleculeEntry):
        """
            Initilizes RedoxReaction.reactant to be in the form of a MolecularEntry,
            RedoxReaction.product to be in the form of MolecularEntry,
            Reaction.reactant to be in the form of a of a list of MolecularEntry of length 1
            Reaction.products to be in the form of a of a list of MolecularEntry of length 1

          Args:
            :param reactant MolecularEntry object
            :param product MolecularEntry object

        """
        self.reactant = reactant
        self.product = product
        self.electron_free_energy = None
        super().__init__([self.reactant], [self.product])

    def graph_representation(self, input_graph) -> nx.DiGraph:
        """
            A method to convert a RedoxReaction class object into graph representation (nx.Digraph object).
            Redox Reaction must be of type 1 reactant -> 1 product

            :return nx.Digraph object of a single Redox Reaction
        """

        return graph_rep_1_1(self, input_graph)

    @classmethod
    def generate(cls, entries: MappingDict) -> List[Reaction]:

        """
        A method to generate all the possible redox reactions from given entries

        Args:
           :param entries: ReactionNetwork(input_entries).entries, entries = {[formula]:{[Nbonds]:{[charge]:MoleculeEntry}}}
           :return list of RedoxReaction class objects
        """

        reactions = []
        for formula in entries:
            for Nbonds in entries[formula]:
                charges = list(entries[formula][Nbonds].keys())
                if len(charges) > 1:
                    for ii in range(len(charges) - 1):
                        charge0 = charges[ii]
                        charge1 = charges[ii + 1]
                        if charge1 - charge0 == 1:
                            for entry0 in entries[formula][Nbonds][charge0]:
                                for entry1 in entries[formula][Nbonds][charge1]:
                                    if entry0.mol_graph.isomorphic_to(entry1.mol_graph):
                                        r = cls(entry0, entry1)
                                        reactions.append(r)

        return reactions

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        """
        A method to identify type of redox reaction (oxidation or reduction)

        Args:
           :return dictionary of the form {"class": "RedoxReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
           where rnx_type_A is the primary type of the reaction based on the reactant and product of the RedoxReaction
           object, and the backwards of this reaction would be rnx_type_B
        """

        if self.product.charge < self.reactant.charge:
            rxn_type_A = "One electron reduction"
            rxn_type_B = "One electron oxidation"
        else:
            rxn_type_A = "One electron oxidation"
            rxn_type_B = "One electron reduction"

        reaction_type = {"class": "RedoxReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self) -> Mapping_Energy_Dict:
        """
           A method to determine the free energy of the redox reaction. Note to set RedoxReaction.eletron_free_energy a value.

           Args:
              :return dictionary of the form {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}
              where free_energy_A is the primary type of the reaction based on the reactant and product of the RedoxReaction
              object, and the backwards of this reaction would be free_energy_B.
        """

        if self.product.free_energy is not None and self.reactant.free_energy is not None:
            free_energy_A = self.product.free_energy - self.reactant.free_energy
            free_energy_B = self.reactant.free_energy - self.product.free_energy

            if self.reaction_type()["rxn_type_A"] == "One electron reduction":
                free_energy_A += -self.electron_free_energy
                free_energy_B += self.electron_free_energy
            else:
                free_energy_A += self.electron_free_energy
                free_energy_B += -self.electron_free_energy
        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
           A method to determine the energy of the redox reaction

           Args:
              :return dictionary of the form {"energy_A": energy_A, "energy_B": energy_B}
              where energy_A is the primary type of the reaction based on the reactant and product of the RedoxReaction
              object, and the backwards of this reaction would be energy_B.
        """
        if self.product.energy is not None and self.reactant.energy is not None:
            energy_A = self.product.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product.energy
        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate(self):
        pass


class IntramolSingleBondChangeReaction(Reaction):
    """
    A class to define intramolecular single bond change as follows:
        Intramolecular formation / breakage of one bond
        A^n <-> B^n
        Two entries with:
            identical composition
            number of edges differ by 1
            identical charge
            removing one of the edges in the graph with more edges yields a graph isomorphic to the other entry

    Args:
       reactant([MolecularEntry]): list of single molecular entry
       product([MoleculeEntry]): list of single molecular entry
    """

    def __init__(self, reactant: MoleculeEntry, product: MoleculeEntry):
        """
            Initilizes IntramolSingleBondChangeReaction.reactant to be in the form of a MolecularEntry,
            IntramolSingleBondChangeReaction.product to be in the form of MolecularEntry,
            Reaction.reactant to be in the form of a of a list of MolecularEntry of length 1
            Reaction.products to be in the form of a of a list of MolecularEntry of length 1

          Args:
            :param reactant MolecularEntry object
            :param product MolecularEntry object

        """

        self.reactant = reactant
        self.product = product
        super().__init__([self.reactant], [self.product])

    def graph_representation(self, input_graph) -> nx.DiGraph:
        """
            A method to convert a IntramolSingleBondChangeReaction class object into graph representation (nx.Digraph object).
           IntramolSingleBondChangeReaction must be of type 1 reactant -> 1 product

            :return nx.Digraph object of a single IntramolSingleBondChangeReaction object
        """

        return graph_rep_1_1(self, input_graph)

    @classmethod
    def generate(cls, entries: MappingDict) -> List[Reaction]:

        """
            A method to generate all the possible intermolecular single bond change reactions from given entries

            Args:
               :param entries: ReactionNetwork(input_entries).entries, entries = {[formula]:{[Nbonds]:{[charge]:MoleculeEntry}}}
               :return list of IntramolSingleBondChangeReaction class objects
        """

        reactions = []
        for formula in entries:
            Nbonds_list = list(entries[formula].keys())
            if len(Nbonds_list) > 1:
                for ii in range(len(Nbonds_list) - 1):
                    Nbonds0 = Nbonds_list[ii]
                    Nbonds1 = Nbonds_list[ii + 1]
                    if Nbonds1 - Nbonds0 == 1:
                        for charge in entries[formula][Nbonds0]:
                            if charge in entries[formula][Nbonds1]:
                                for entry1 in entries[formula][Nbonds1][charge]:
                                    for edge in entry1.edges:
                                        mg = copy.deepcopy(entry1.mol_graph)
                                        mg.break_edge(edge[0], edge[1], allow_reverse=True)
                                        if nx.is_weakly_connected(mg.graph):
                                            for entry0 in entries[formula][Nbonds0][charge]:
                                                if entry0.mol_graph.isomorphic_to(mg):
                                                    r = cls(entry0, entry1)
                                                    reactions.append(r)
                                                    break

        return reactions

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        """
            A method to identify type of intramolecular single bond change reaction (bond breakage or formation)

            Args:
               :return dictionary of the form {"class": "IntramolSingleBondChangeReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
               where rnx_type_A is the primary type of the reaction based on the reactant and product of the IntramolSingleBondChangeReaction
               object, and the backwards of this reaction would be rnx_type_B
        """
        if self.product.charge < self.reactant.charge:
            rxn_type_A = "Intramolecular single bond breakage"
            rxn_type_B = "Intramolecular single bond formation"
        else:
            rxn_type_A = "Intramolecular single bond formation"
            rxn_type_B = "Intramolecular single bond breakage"

        reaction_type = {"class": "IntramolSingleBondChangeReaction", "rxn_type_A": rxn_type_A,
                         "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self) -> Mapping_Energy_Dict:
        """
          A method to  determine the free energy of the intramolecular single bond change reaction

          Args:
             :return dictionary of the form {"free_energy_A": energy_A, "free_energy_B": energy_B}
             where free_energy_A is the primary type of the reaction based on the reactant and product of the IntramolSingleBondChangeReaction
             object, and the backwards of this reaction would be free_energy_B.
        """
        if self.product.free_energy is not None and self.reactant.free_energy is not None:
            free_energy_A = self.product.free_energy - self.reactant.free_energy
            free_energy_B = self.reactant.free_energy - self.product.free_energy
        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
          A method to determine the energy of the intramolecular single bond change reaction

          Args:
             :return dictionary of the form {"energy_A": energy_A, "energy_B": energy_B}
             where energy_A is the primary type of the reaction based on the reactant and product of the IntramolSingleBondChangeReaction
             object, and the backwards of this reaction would be energy_B.
         """

        if self.product.energy is not None and self.reactant.energy is not None:
            energy_A = self.product.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product.energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate(self):
        pass


class IntermolecularReaction(Reaction):
    """
        A class to define intermolecular bond change as follows:
            Intermolecular formation / breakage of one bond
            A <-> B + C aka B + C <-> A
            Three entries with:
                comp(A) = comp(B) + comp(C)
                charge(A) = charge(B) + charge(C)
                removing one of the edges in A yields two disconnected subgraphs that are isomorphic to B and C

        Args:
           reactant([MolecularEntry]): list of single molecular entry
           product([MoleculeEntry]): list of two molecular entries
    """

    def __init__(self, reactant: MoleculeEntry, product: List[MoleculeEntry]):
        """
            Initilizes IntermolecularReaction.reactant to be in the form of a MolecularEntry,
            IntermolecularReaction.product to be in the form of [MolecularEntry_0, MolecularEntry_1],
            Reaction.reactant to be in the form of a of a list of MolecularEntry of length 1
            Reaction.products to be in the form of a of a list of MolecularEntry of length 2

          Args:
            :param reactant MolecularEntry object
            :param product list of MolecularEntry object of length 2

        """

        self.reactant = reactant
        self.product_0 = product[0]
        self.product_1 = product[1]
        super().__init__([self.reactant], [self.product_0, self.product_1])

    def graph_representation(self, input_graph) -> nx.DiGraph:

        """
            A method to convert a IntermolecularReaction class object into graph representation (nx.Digraph object).
            IntermolecularReaction must be of type 1 reactant -> 2 products

            :return nx.Digraph object of a single IntermolecularReaction object
        """

        return graph_rep_1_2(self, input_graph)

    @classmethod
    def generate(cls, entries: MappingDict) -> List[Reaction]:

        """
           A method to generate all the possible intermolecular reactions from given entries

           Args:
              :param entries: ReactionNetwork(input_entries).entries, entries = {[formula]:{[Nbonds]:{[charge]:MoleculeEntry}}}
              :return list of IntermolecularReaction class objects
        """
        reactions = []
        for formula in entries:
            for Nbonds in entries[formula]:
                if Nbonds > 0:
                    for charge in entries[formula][Nbonds]:
                        for entry in entries[formula][Nbonds][charge]:
                            for edge in entry.edges:
                                bond = [(edge[0], edge[1])]
                                try:
                                    frags = entry.mol_graph.split_molecule_subgraphs(bond, allow_reverse=True)
                                    formula0 = frags[0].molecule.composition.alphabetical_formula
                                    Nbonds0 = len(frags[0].graph.edges())
                                    formula1 = frags[1].molecule.composition.alphabetical_formula
                                    Nbonds1 = len(frags[1].graph.edges())
                                    if formula0 in entries and formula1 in entries:
                                        if Nbonds0 in entries[formula0] and Nbonds1 in entries[formula1]:
                                            for charge0 in entries[formula0][Nbonds0]:
                                                for entry0 in entries[formula0][Nbonds0][charge0]:
                                                    if frags[0].isomorphic_to(entry0.mol_graph):
                                                        charge1 = charge - charge0
                                                        if charge1 in entries[formula1][Nbonds1]:
                                                            for entry1 in entries[formula1][Nbonds1][charge1]:
                                                                if frags[1].isomorphic_to(entry1.mol_graph):
                                                                    # r1 = ReactionEntry([entry], [entry0, entry1])
                                                                    r = cls(entry, [entry0, entry1])
                                                                    reactions.append(r)
                                                                    break
                                                        break
                                except MolGraphSplitError:
                                    pass

        return reactions

    def reaction_type(self) -> Mapping_ReactionType_Dict:

        """
           A method to identify type of intermoleular reaction (bond decomposition from one to two or formation from two to one molecules)

           Args:
              :return dictionary of the form {"class": "IntermolecularReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
              where rnx_type_A is the primary type of the reaction based on the reactant and product of the IntermolecularReaction
              object, and the backwards of this reaction would be rnx_type_B
        """

        rxn_type_A = "Molecular decomposition breaking one bond A -> B+C"
        rxn_type_B = "Molecular formation from one new bond A+B -> C"

        reaction_type = {"class": "IntermolecularReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self) -> Mapping_Energy_Dict:
        """
          A method to determine the free energy of the intermolecular reaction

          Args:
             :return dictionary of the form {"free_energy_A": energy_A, "free_energy_B": energy_B}
             where free_energy_A is the primary type of the reaction based on the reactant and product of the IntermolecularReaction
             object, and the backwards of this reaction would be free_energy_B.
         """
        if self.product_1.free_energy is not None and self.product_0.free_energy is not None and self.reactant.free_energy is not None:
            free_energy_A = self.product_0.free_energy + self.product_1.free_energy - self.reactant.free_energy
            free_energy_B = self.reactant.free_energy - self.product_0.free_energy - self.product_1.free_energy

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
          A method to determine the energy of the intermolecular reaction

          Args:
             :return dictionary of the form {"energy_A": energy_A, "energy_B": energy_B}
             where energy_A is the primary type of the reaction based on the reactant and product of the IntermolecularReaction
             object, and the backwards of this reaction would be energy_B.
        """
        if self.product_1.energy is not None and self.product_0.energy is not None and self.reactant.energy is not None:
            energy_A = self.product_0.energy + self.product_1.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product_0.energy - self.product_1.energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate(self):
        pass


class CoordinationBondChangeReaction(Reaction):
    """
    A class to define coordination bond change as follows:
        Simultaneous formation / breakage of multiple coordination bonds
        A + M <-> AM aka AM <-> A + M
        Three entries with:
            M = Li or Mg
            comp(AM) = comp(A) + comp(M)
            charge(AM) = charge(A) + charge(M)
            removing two M-containing edges in AM yields two disconnected subgraphs that are isomorphic to B and C

    Args:
       reactant([MolecularEntry]): list of single molecular entry
       product([MoleculeEntry]): list of two molecular entries
    """

    def __init__(self, reactant: MoleculeEntry, product: List[MoleculeEntry]):
        """
            Initilizes CoordinationBondChangeReaction.reactant to be in the form of a MolecularEntry,
            CoordinationBondChangeReaction.product to be in the form of [MolecularEntry_0, MolecularEntry_1],
            Reaction.reactant to be in the form of a of a list of MolecularEntry of length 1
            Reaction.products to be in the form of a of a list of MolecularEntry of length 2

          Args:
            :param reactant MolecularEntry object
            :param product list of MolecularEntry object of length 2

        """
        self.reactant = reactant
        self.product_0 = product[0]
        self.product_1 = product[1]
        super().__init__([self.reactant], [self.product_0, self.product_1])

    def graph_representation(self, input_graph) -> nx.DiGraph:
        """
            A method to convert a CoordinationBondChangeReaction class object into graph representation (nx.Digraph object).
            CoordinationBondChangeReaction must be of type 1 reactant -> 2 products

            :return nx.Digraph object of a single CoordinationBondChangeReaction object
        """

        return graph_rep_1_2(self, input_graph)

    @classmethod
    def generate(cls, entries: MappingDict) -> List[Reaction]:
        """
          A method to generate all the possible coordination bond chamge reactions from given entries

          Args:
             :param entries: ReactionNetwork(input_entries).entries, entries = {[formula]:{[Nbonds]:{[charge]:MoleculeEntry}}}
             :return list of CoordinationBondChangeReaction class objects
        """
        reactions = []
        M_entries = {}
        for formula in entries:
            if formula == "Li1" or formula == "Mg1":
                if formula not in M_entries:
                    M_entries[formula] = {}
                for charge in entries[formula][0]:
                    assert (len(entries[formula][0][charge]) == 1)
                    M_entries[formula][charge] = entries[formula][0][charge][0]
        if M_entries != {}:
            for formula in entries:
                if "Li" in formula or "Mg" in formula:
                    for Nbonds in entries[formula]:
                        if Nbonds > 2:
                            for charge in entries[formula][Nbonds]:
                                for entry in entries[formula][Nbonds][charge]:
                                    nosplit_M_bonds = []
                                    for edge in entry.edges:
                                        if str(entry.molecule.sites[edge[0]].species) in M_entries or str(
                                                entry.molecule.sites[edge[1]].species) in M_entries:
                                            M_bond = (edge[0], edge[1])
                                            try:
                                                frags = entry.mol_graph.split_molecule_subgraphs([M_bond],
                                                                                                 allow_reverse=True)
                                            except MolGraphSplitError:
                                                nosplit_M_bonds.append(M_bond)
                                    bond_pairs = itertools.combinations(nosplit_M_bonds, 2)
                                    for bond_pair in bond_pairs:
                                        try:
                                            frags = entry.mol_graph.split_molecule_subgraphs(bond_pair,
                                                                                             allow_reverse=True)
                                            M_ind = None
                                            M_formula = None
                                            for ii, frag in enumerate(frags):
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
                                                        if nonM_formula in entries:
                                                            if nonM_Nbonds in entries[nonM_formula]:
                                                                for nonM_charge in entries[nonM_formula][
                                                                    nonM_Nbonds]:
                                                                    M_charge = entry.charge - nonM_charge
                                                                    if M_charge in M_entries[M_formula]:
                                                                        for nonM_entry in \
                                                                                entries[nonM_formula][nonM_Nbonds][
                                                                                    nonM_charge]:
                                                                            if frag.isomorphic_to(nonM_entry.mol_graph):
                                                                                r = cls(entry, [nonM_entry,
                                                                                                M_entries[
                                                                                                    M_formula][
                                                                                                    M_charge]])
                                                                                reactions.append(r)
                                                                                break
                                        except MolGraphSplitError:
                                            pass
        return reactions

    def reaction_type(self) -> Mapping_ReactionType_Dict:
        """
           A method to identify type of coordination bond change reaction (bond breaking from one to two or forming from two to one molecules)

           Args:
              :return dictionary of the form {"class": "CoordinationBondChangeReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
              where rnx_type_A is the primary type of the reaction based on the reactant and product of the CoordinationBondChangeReaction
              object, and the backwards of this reaction would be rnx_type_B
        """

        rxn_type_A = "Coordination bond breaking AM -> A+M"
        rxn_type_B = "Coordination bond forming A+M -> AM"

        reaction_type = {"class": "CoordinationBondChangeReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self) -> Mapping_Energy_Dict:
        """
              A method to determine the free energy of the coordination bond chnage reaction

              Args:
                 :return dictionary of the form {"free_energy_A": energy_A, "free_energy_B": energy_B}
                 where free_energy_A is the primary type of the reaction based on the reactant and product of the CoordinationBondChangeReaction
                 object, and the backwards of this reaction would be free_energy_B.
         """
        if self.product_1.free_energy is not None and self.product_0.free_energy is not None and self.reactant.free_energy is not None:
            free_energy_A = self.product_0.free_energy + self.product_1.free_energy - self.reactant.free_energy
            free_energy_B = self.reactant.free_energy - self.product_0.free_energy - self.product_1.free_energy

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
              A method to determine the energy of the coordination bond change reaction

              Args:
                 :return dictionary of the form {"energy_A": energy_A, "energy_B": energy_B}
                 where energy_A is the primary type of the reaction based on the reactant and product of the CoordinationBondChangeReaction
                 object, and the backwards of this reaction would be energy_B.
        """
        if self.product_1.energy is not None and self.product_0.energy is not None and self.reactant.energy is not None:
            energy_A = self.product_0.energy + self.product_1.energy - self.reactant.energy
            energy_B = self.reactant.energy - self.product_0.energy - self.product_1.energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate(self):
        pass


class ConcertedReaction(Reaction):
    """
        A class to define concerted reactions.
        User can specify either allowing <=1 bond breakage + <=1 bond formation OR <=2 bond breakage + <=2 bond formation.
        User can also specify how many electrons are allowed to involve in a reaction.
        Can only deal with <= 2 reactants and <=2 products for now.
        For 1 reactant -> 1 product reactions, a maximum 1 bond breakage and 1 bond formation is allowed,
        even when the user specify "<=2 bond breakage + <=2 bond formation".
        Args:
           reactant([MolecularEntry]): list of 1-2 molecular entries
           product([MoleculeEntry]): list of 1-2 molecular entries
    """

    def __init__(self, reactant: List[MoleculeEntry], product: List[MoleculeEntry]):
        """
            Initilizes IntermolecularReaction.reactant to be in the form of a MolecularEntry,
            IntermolecularReaction.product to be in the form of [MolecularEntry_0, MolecularEntry_1],
            Reaction.reactant to be in the form of a of a list of MolecularEntry of length 1
            Reaction.products to be in the form of a of a list of MolecularEntry of length 2
          Args:
            :param reactant MolecularEntry object
            :param product list of MolecularEntry object of length 2
        """

        self.reactants = reactant
        self.products = product
        self.electron_free_energy = None
        self.electron_energy = None
        super().__init__(reactant, product)

    def graph_representation(self) -> nx.DiGraph:  # temp here, use graph_rep_1_2 instead

        """
            A method to convert a Concerted class object into graph representation (nx.Digraph object).
            IntermolecularReaction must be of type 1 reactant -> 2 products
            :return nx.Digraph object of a single IntermolecularReaction object
        """
        if len(self.reactants) == len(self.products) == 1:
            return graph_rep_1_1(self)
        elif len(self.reactants) == 1 and len(self.products) == 2:
            return graph_rep_1_2(self)
        elif len(self.reactants) == 2 and len(self.products) == 1:
            self.reactants, self.products = self.products, self.reactants
            return graph_rep_1_2(self)
        elif len(self.reactants) == len(self.products) == 2:
            return graph_rep_2_2(self)

    @classmethod
    def generate(cls, entries_list: [MoleculeEntry], name="nothing", read_file=False, num_processors=16, reaction_type="break2_form2", allowed_charge_change=0) -> List[Reaction]:

        """
           A method to generate all the possible concerted reactions from given entries_list.
           Args:
              :param entries_list, entries_list = [MoleculeEntry]
              :param name(str): The name to put in FindConcertedReactions class. For reading in the files generated from that class.
              :param read_file(bool): whether to read in the file generated from the FindConcertedReactions class.
                                     If true, name+'_concerted_rxns.json' has to be present in the running directory.
                                     If False, will find concerted reactions on the fly.
                                     Note that this will take a couple hours when running on 16 CPU with < 100 entries.
              :param num_processors:
              :param reaction_type: Can choose from "break2_form2" and "break1_form1"
              :param allowed_charge_change: How many charge changes are allowed in a concerted reaction.
                          If zero, sum(reactant total charges) = sun(product total charges). If n(non-zero), allow n-electron redox reactions.
              :return list of IntermolecularReaction class objects
        """
        if read_file:
            all_concerted_reactions = loadfn(name+'_concerted_rxns.json')
        else:
            from pymatgen.analysis.reaction_network.extract_reactions import FindConcertedReactions
            FCR = FindConcertedReactions(entries_list, name)
            all_concerted_reactions = FCR.get_final_concerted_reactions(name, num_processors, reaction_type)

        reactions = []
        for reaction in all_concerted_reactions:
            reactants = reaction[0].split("_")
            products = reaction[1].split("_")
            entries0 = [entries_list[int(item)] for item in reactants]
            entries1 = [entries_list[int(item)] for item in products]
            reactant_total_charge = np.sum([item.charge for item in entries0])
            product_total_charge = np.sum([item.charge for item in entries1])
            total_charge_change = product_total_charge - reactant_total_charge
            if abs(total_charge_change) <= allowed_charge_change:
                r = cls(entries0,entries1)
                reactions.append(r)

        return reactions

    def reaction_type(self) -> Mapping_ReactionType_Dict:

        """
           A method to identify type of intermoleular reaction (bond decomposition from one to two or formation from two to one molecules)
           Args:
              :return dictionary of the form {"class": "IntermolecularReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
              where rnx_type_A is the primary type of the reaction based on the reactant and product of the IntermolecularReaction
              object, and the backwards of this reaction would be rnx_type_B
        """

        rxn_type_A = "Concerted"
        rxn_type_B = "Concerted"

        reaction_type = {"class": "ConcertedReaction", "rxn_type_A": rxn_type_A, "rxn_type_B": rxn_type_B}
        return reaction_type

    def free_energy(self) -> Mapping_Energy_Dict:
        """
          A method to determine the free energy of the concerted reaction
          Args:
             :return dictionary of the form {"free_energy_A": energy_A, "free_energy_B": energy_B}
             where free_energy_A is the primary type of the reaction based on the reactant and product of the ConcertedReaction
             object, and the backwards of this reaction would be free_energy_B.
         """
        if all(reactant.free_energy != None for reactant in self.reactants) and all(product.free_energy != None for product in self.products):
            reactant_total_charge = np.sum([item.charge for item in self.reactants])
            product_total_charge = np.sum([item.charge for item in self.products])
            reactant_total_free_energy = np.sum([item.free_energy for item in self.reactants])
            product_total_free_energy = np.sum([item.free_energy for item in self.products])
            total_charge_change = product_total_charge - reactant_total_charge
            free_energy_A = product_total_free_energy - reactant_total_free_energy + total_charge_change * self.electron_free_energy
            free_energy_B = reactant_total_free_energy - product_total_free_energy - total_charge_change * self.electron_free_energy

        else:
            free_energy_A = None
            free_energy_B = None

        return {"free_energy_A": free_energy_A, "free_energy_B": free_energy_B}

    def energy(self) -> Mapping_Energy_Dict:
        """
          A method to determine the energy of the concerted reaction
          Args:
             :return dictionary of the form {"energy_A": energy_A, "energy_B": energy_B}
             where energy_A is the primary type of the reaction based on the reactant and product of the ConcertedReaction
             object, and the backwards of this reaction would be energy_B.
             Electron electronic energy set to 0 for now.
        """
        if all(reactant.energy != None for reactant in self.reactants) and all(
                product.energy != None for product in self.products):
            reactant_total_charge = np.sum([item.charge for item in self.reactants])
            product_total_charge = np.sum([item.charge for item in self.products])
            reactant_total_energy = np.sum([item.energy for item in self.reactants])
            product_total_energy = np.sum([item.energy for item in self.products])
            total_charge_change = product_total_charge - reactant_total_charge
            energy_A = product_total_energy - reactant_total_energy #+ total_charge_change * self.electron_energy
            energy_B = reactant_total_energy - product_total_energy #- total_charge_change * self.electron_energy

        else:
            energy_A = None
            energy_B = None

        return {"energy_A": energy_A, "energy_B": energy_B}

    def rate(self):
        pass


class ReactionPath(MSONable):
    """
        A class to define path object within the reaction network which constains all the associated characteristic attributes of a given path

        :param path - a list of nodes that defines a path from node A to B within a graph built using ReactionNetwork.build()
    """

    def __init__(self, path):
        """
        initializes the ReactionPath object attributes for a given path
        :param path: a list of nodes that defines a path from node A to B within a graph built using ReactionNetwork.build()
        """

        self.path = path
        self.byproducts = []
        self.unsolved_prereqs = []
        self.solved_prereqs = []
        self.all_prereqs = []
        self.cost = 0.0
        self.overall_free_energy_change = 0.0
        self.hardest_step = None
        self.description = ""
        self.pure_cost = 0.0
        self.full_path = None
        self.hardest_step_deltaG = None
        self.path_dict = {"byproducts": self.byproducts, "unsolved_prereqs": self.unsolved_prereqs,
                          "solved_prereqs": self.solved_prereqs, "all_prereqs": self.all_prereqs, "cost": self.cost,
                          "path": self.path, "overall_free_energy_change": self.overall_free_energy_change,
                          "hardest_step": self.hardest_step, "description": self.description,
                          "pure_cost": self.pure_cost,
                          "hardest_step_deltaG": self.hardest_step_deltaG, "full_path": self.full_path}

    @property
    def as_dict(self) -> dict:
        """
            A method to convert ReactionPath objection into a dictionary
        :return: d: dictionary containing all te ReactionPath attributes
        """
        d = {"@module": self.__class__.__module__,
             "@class": self.__class__.__name__,
             "byproducts": self.byproducts,
             "unsolved_prereqs": self.unsolved_prereqs,
             "solved_prereqs": self.solved_prereqs,
             "all_prereqs": self.all_prereqs,
             "cost": self.cost,
             "path": self.path,
             "overall_free_energy_change": self.overall_free_energy_change,
             "hardest_step": self.hardest_step,
             "description": self.description,
             "pure_cost": self.pure_cost,
             "hardest_step_deltaG": self.hardest_step_deltaG,
             "full_path": self.full_path,
             "path_dict": self.path_dict
             }
        return d

    @classmethod
    def from_dict(cls, d):
        """
            A method to convert dict to ReactionPath object
        :param d:  dict retuend from ReactionPath.as_dict() method
        :return: ReactionPath object
        """
        x = cls(d.get("path"))
        x.byproducts = d.get("byproducts")
        x.unsolved_prereqs = d.get("unsolved_prereqs")
        x.solved_prereqs = d.get("solved_prereqs")
        x.all_prereqs = d.get("all_prereqs")
        x.cost = d.get("cost", 0)

        x.overall_free_energy_change = d.get("overall_free_energy_change", 0)
        x.hardest_step = d.get("hardest_step")
        x.description = d.get("description")
        x.pure_cost = d.get("pure_cost", 0)
        x.hardest_step_deltaG = d.get("hardest_step_deltaG")
        x.full_path = d.get("full_path")
        x.path_dict = d.get("path_dict")

        return x

    @classmethod
    def characterize_path(cls, path, weight: str, min_cost: Dict[str, float], input_graph: GraphTool,
                          old_solved_PRs=[], PR_byproduct_dict = {}, actualPRs = {}):  # -> ReactionPath
        """
            A method to define ReactionPath attributes based on the inputs
        :param path: a list of nodes that defines a path from node A to B within a graph built using ReactionNetwork.build()
        :param weight: string (either "softplus" or "exponent")
        :param min_cost: dict with minimum cost from path start to a node, of from {node: float}
        :param graph: nx.Digraph
        :param PR_paths: list of already solved PRs
        :return: ReactionPath object
        """
        if isinstance(weight, str):
            weight = eval('input_graph.' + weight)

        if path is None:
            class_instance = cls(None)
        else:
            #path = []
            v_map = []
            for node in path:
                v_map.append(int(input_graph.vlabel_to_vind_dict[str(node)]))
            # for v in vlist:
            #     v_label = input_graph.node_label[int(v)]
            #     if v_label.isdigit():
            #         path.append(int(input_graph.node_label[int(v)]))
            #     else:
            #         path.append(input_graph.node_label[int(v)])
            #cost = 0.0
            # #w = weight
            # w = eval("input_graph."+weight)
            # for e in elist:
            #     cost = cost + w[e]
            class_instance = cls(path)
            #class_instance.cost = cost
            # # if path[-1] == 9:
            # #     print("^^", path, class_instance.cost)

            pool = []
            pool.append(int(path[0]))
            for ii, step in enumerate(path):
                if ii != len(path) - 1:
                    # print(v_map)
                    # print(v_map[ii])
                    # input_graph.graph.edge(456, 10358)

                    class_instance.cost += weight[input_graph.graph.edge(v_map[ii], v_map[ii + 1])]
                    if ii % 2 == 1:
                        rxn = step.split(",")
                        if "+PR_" in rxn[0]:
                            a = int(rxn[0].split("+PR_")[0])
                            PR_b = int(rxn[0].split("+PR_")[1])
                            c = int(rxn[1])
                            pool_modified = copy.deepcopy(pool)
                            pool_modified.remove(int(a))
                            if PR_b in pool_modified:
                                class_instance.cost = class_instance.cost - min_cost[PR_b]
                                pool.remove(int(a))
                                pool.remove(int(PR_b))
                                pool.append(int(c))
                            elif PR_b not in pool_modified:
                                if PR_b in old_solved_PRs:
                                    # if PR is solved
                                    class_instance.solved_prereqs.append(PR_b)
                                    class_instance.all_prereqs.append(PR_b)
                                    PR_b_byproducts = PR_byproduct_dict[PR_b]["byproducts"]
                                    start = int(PR_byproduct_dict[PR_b]["start"])
                                    if a in PR_b_byproducts:
                                        new_path_piece1 = actualPRs[PR_b][start].path
                                        new_path_piece2 = [str(PR_b)+"+"+"PR_"+str(a)+","+str(c)]
                                        new_path_piece3 = path[ii+1::]
                                        new_path = new_path_piece1+new_path_piece2+new_path_piece3
                                        assert(int(c) == int(path[ii+1]))
                                        return ReactionPath.characterize_path(new_path, weight, min_cost, input_graph, old_solved_PRs, PR_byproduct_dict, actualPRs)
                                    elif a not in PR_b_byproducts:
                                        pool.remove(int(a))
                                        pool = pool+PR_b_byproducts
                                        pool.append(int(c))
                                elif PR_b not in old_solved_PRs:
                                    class_instance.unsolved_prereqs.append(PR_b)
                                    class_instance.all_prereqs.append(PR_b)
                                    pool.remove(int(a))
                                    pool.append(int(c))
                        elif "+" in rxn[1]:
                            # node = A,B+C
                            a = int(rxn[0])
                            b = int(rxn[1].split("+")[0])
                            c = int(rxn[1].split("+")[1])
                            pool.remove(int(a))
                            pool.append(int(b))
                            pool.append(int(c))
                        else:
                            # node = A,B
                            a = int(rxn[0])
                            b = int(rxn[1])
                            pool.remove(int(a))
                            pool.append(int(b))

            pool.remove(int(path[-1]))
            class_instance.byproducts = pool


            # for ii, step in enumerate(path):
            #     if ii != len(path) - 1:
            #         # print(v_map[ii], v_map[ii + 1], input_graph.graph.edge(v_map[ii], v_map[ii+1]))
            #         # print(type(weight))
            #         # print(weight[input_graph.graph.edge(v_map[ii], v_map[ii+1])])
            #         class_instance.cost += weight[input_graph.graph.edge(v_map[ii], v_map[ii+1])]
            #             #graph[step][path[ii + 1]][weight]
            #         if ii % 2 == 1:
            #             rxn = step.split(",")
            #             if "+PR_" in rxn[0]:
            #                 PR = int(rxn[0].split("+PR_")[1])
            #                 class_instance.all_prereqs.append(PR)
            #             if "+" in rxn[1]:
            #                 desired_prod_satisfied = False
            #                 prods = rxn[1].split("+")
            #                 for prod in prods:
            #                     if int(prod) != path[ii + 1]:
            #                         class_instance.byproducts.append(int(prod))
            #                     elif desired_prod_satisfied:
            #                         class_instance.byproducts.append(int(prod))
            #                     else:
            #                         desired_prod_satisfied = True
            # for PR in class_instance.all_prereqs:
            #     if PR in class_instance.byproducts:
            #         # Note that we're ignoring the order in which BPs are made vs they come up as PRs...
            #         class_instance.all_prereqs.remove(PR)
            #         class_instance.byproducts.remove(PR)
            #
            #         if PR in min_cost:
            #             class_instance.cost -= min_cost[PR]
            #         else:
            #             print("Missing PR cost to remove:", PR)
            # for PR in class_instance.all_prereqs:
            #     if PR in PR_paths:
            #         class_instance.solved_prereqs.append(PR)
            #     else:
            #         class_instance.unsolved_prereqs.append(PR)

            class_instance.path_dict = {"byproducts": class_instance.byproducts,
                                        "unsolved_prereqs": class_instance.unsolved_prereqs,
                                        "solved_prereqs": class_instance.solved_prereqs,
                                        "all_prereqs": class_instance.all_prereqs, "cost": class_instance.cost,
                                        "path": class_instance.path,
                                        "overall_free_energy_change": class_instance.overall_free_energy_change,
                                        "hardest_step": class_instance.hardest_step,
                                        "description": class_instance.description,
                                        "pure_cost": class_instance.pure_cost,
                                        "hardest_step_deltaG": class_instance.hardest_step_deltaG,
                                        "full_path": class_instance.full_path}
        return class_instance

    # @classmethod
    # def characterize_path(cls, path: List[str], weight: str, min_cost: Dict[str, float], graph: nx.DiGraph,
    #                       old_solved_PRs=[], PR_byproduct_dict={}, actualPRs={}):  # -> ReactionPath
    #     """
    #         A method to define ReactionPath attributes based on the inputs
    #     :param path: a list of nodes that defines a path from node A to B within a graph built using ReactionNetwork.build()
    #     :param weight: string (either "softplus" or "exponent")
    #     :param min_cost: dict with minimum cost from path start to a node, of from {node: float}
    #     :param graph: nx.Digraph
    #     :param PR_paths: list of already solved PRs
    #     :return: ReactionPath object
    #     """
    #     if path is None:
    #         class_instance = cls(None)
    #     else:
    #         class_instance = cls(path)
    #         pool = []
    #         pool.append(path[0])
    #         for ii, step in enumerate(path):
    #             if ii != len(path) - 1:
    #                 class_instance.cost += graph[step][path[ii + 1]][weight]
    #                 if ii % 2 == 1:
    #                     rxn = step.split(",")
    #                     if "+PR_" in rxn[0]:
    #                         a = int(rxn[0].split("+PR_")[0])
    #                         PR_b = int(rxn[0].split("+PR_")[1])
    #                         c = int(rxn[1])
    #                         pool_modified = copy.deepcopy(pool)
    #                         pool_modified.remove(a)
    #                         if PR_b in pool_modified:
    #                             class_instance.cost = class_instance.cost - min_cost[PR_b]
    #                             pool.remove(a)
    #                             pool.remove(PR_b)
    #                             pool.append(c)
    #                         elif PR_b not in pool_modified:
    #                             if PR_b in old_solved_PRs:
    #                                 # if PR is solved
    #                                 class_instance.solved_prereqs.append(PR_b)
    #                                 class_instance.all_prereqs.append(PR_b)
    #                                 PR_b_byproducts = PR_byproduct_dict[PR_b]["byproducts"]
    #                                 start = int(PR_byproduct_dict[PR_b]["start"])
    #                                 if a in PR_b_byproducts:
    #                                     new_path_piece1 = actualPRs[PR_b][start].path
    #                                     new_path_piece2 = [str(PR_b) + "+" + "PR_" + str(a) + "," + str(c)]
    #                                     new_path_piece3 = path[ii + 1::]
    #                                     new_path = new_path_piece1 + new_path_piece2 + new_path_piece3
    #                                     assert (c == path[ii + 1])
    #                                     return ReactionPath.characterize_path(new_path, weight, min_cost, graph,
    #                                                                           old_solved_PRs, PR_byproduct_dict,
    #                                                                           actualPRs)
    #                                 elif a not in PR_b_byproducts:
    #                                     pool.remove(a)
    #                                     pool = pool + PR_b_byproducts
    #                                     pool.append(c)
    #                             elif PR_b not in old_solved_PRs:
    #                                 class_instance.unsolved_prereqs.append(PR_b)
    #                                 class_instance.all_prereqs.append(PR_b)
    #                                 pool.remove(a)
    #                                 pool.append(c)
    #                     elif "+" in rxn[1]:
    #                         # node = A,B+C
    #                         a = int(rxn[0])
    #                         b = int(rxn[1].split("+")[0])
    #                         c = int(rxn[1].split("+")[1])
    #                         pool.remove(a)
    #                         pool.append(b)
    #                         pool.append(c)
    #                     else:
    #                         # node = A,B
    #                         a = int(rxn[0])
    #                         b = int(rxn[1])
    #                         pool.remove(a)
    #                         pool.append(b)
    #
    #         pool.remove(path[-1])
    #         class_instance.byproducts = pool
    #         class_instance.path_dict = {"byproducts": class_instance.byproducts,
    #                                     "unsolved_prereqs": class_instance.unsolved_prereqs,
    #                                     "solved_prereqs": class_instance.solved_prereqs,
    #                                     "all_prereqs": class_instance.all_prereqs, "cost": class_instance.cost,
    #                                     "path": class_instance.path,
    #                                     "overall_free_energy_change": class_instance.overall_free_energy_change,
    #                                     "hardest_step": class_instance.hardest_step,
    #                                     "description": class_instance.description,
    #                                     "pure_cost": class_instance.pure_cost,
    #                                     "hardest_step_deltaG": class_instance.hardest_step_deltaG,
    #                                     "full_path": class_instance.full_path}
    #     # print(class_instance.path_dict)
    #     return class_instance

    @classmethod
    def characterize_path_final(cls, path, weight: str, min_cost: Dict[str, float], input_graph: GraphTool,
                                old_solved_PRs=[], PR_byproduct_dict = {}, PR_paths = {}):  # Mapping_PR_Dict): -> ReactionPath
        """
            A method to define all the attributes of a given path once all the PRs are solved
        :param path: a list of nodes that defines a path from node A to B within a graph built using ReactionNetwork.build()
        :param weight: string (either "softplus" or "exponent")
        :param min_cost: dict with minimum cost from path start to a node, of from {node: float},
        if no path exist, value is "no_path", if path is unsolved yet, value is "unsolved_path"
        :param graph: nx.Digraph
        :param PR_paths: dict that defines a path from each node to a start,
               of the form {int(node1): {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}}, int(node2):...}
        :return: ReactionPath object
        """
        #print(weight.key_type(),weight.value_type(), type(weight))
        #print(weight.__name__)
        #graph_tool.EdgePropertyMap.value_type()
        vind_to_vlabel_dict = {int(n): input_graph.node_label[n] for n in input_graph.graph.vertices()}
        vlabel_to_vind_dict = {input_graph.node_label[n]:int(n) for n in input_graph.graph.vertices()}
        if path is None:
            class_instance = cls(None)
        else:
            class_instance = cls.characterize_path(path, weight, min_cost, input_graph, old_solved_PRs,
                                                   PR_byproduct_dict,
                                                   PR_paths)
            assert (len(class_instance.solved_prereqs) == len(class_instance.all_prereqs))
            assert (len(class_instance.unsolved_prereqs) == 0)

            PRs_to_join = copy.deepcopy(class_instance.all_prereqs)
            full_path = copy.deepcopy(class_instance.path)
            while len(PRs_to_join) > 0:
                new_PRs = []
                for PR in PRs_to_join:
                    PR_path = None
                    PR_min_cost = 1000000000000000.0
                    for start in PR_paths[PR]:
                        if PR_paths[PR][start].path != None:
                            # print(PR_paths[PR][start].path_dict)
                            # print(PR_paths[PR][start].cost, PR_paths[PR][start].overall_free_energy_change, PR_paths[PR][start].path)
                            if PR_paths[PR][start].cost < PR_min_cost:
                                PR_min_cost = PR_paths[PR][start].cost
                                PR_path = PR_paths[PR][start]
                                assert (len(PR_path.solved_prereqs) == len(PR_path.all_prereqs))
                    for new_PR in PR_path.all_prereqs:
                        new_PRs.append(new_PR)
                        class_instance.all_prereqs.append(new_PR)
                    for new_BP in PR_path.byproducts:
                        class_instance.byproducts.append(new_BP)
                    full_path = PR_path.path + full_path
                PRs_to_join = copy.deepcopy(new_PRs)

            for PR in class_instance.all_prereqs:
                if PR in class_instance.byproducts:
                    print("WARNING: Matching prereq and byproduct found!", PR)

            #indices_0 = [i for i, x in enumerate(input_graph.bipartite.a) if x == 1]

            full_path_ind = []
            for node in full_path:
                full_path_ind.append(vlabel_to_vind_dict[str(node)])
            for ii, step in enumerate(full_path_ind):
                if input_graph.bipartite[step] == 1:
                #if graph.nodes[step]["bipartite"] == 1:

                    #class_instance.pure_cost += ReactionNetwork.
                    if weight == "softplus":
                        class_instance.pure_cost += ReactionNetwork.softplus(input_graph.free_energy[step])#   graph.nodes[step]["free_energy"])
                    elif weight == "exponent":
                        class_instance.pure_cost += ReactionNetwork.exponent(input_graph.free_energy[step])

                    class_instance.overall_free_energy_change += input_graph.free_energy[step]

                    if class_instance.description == "":
                        class_instance.description += input_graph.rxn_type[step]
                    else:
                        class_instance.description += ", " + input_graph.rxn_type[step]

                    if class_instance.hardest_step is None:
                        class_instance.hardest_step = vind_to_vlabel_dict[step]
                    elif input_graph.free_energy[step] > input_graph.free_energy[vlabel_to_vind_dict[class_instance.hardest_step]]: # graph.nodes[class_instance.hardest_step]["free_energy"]:
                        class_instance.hardest_step = vind_to_vlabel_dict[step]

            class_instance.full_path = full_path

            if class_instance.hardest_step is None:
                class_instance.hardest_step_deltaG = None
            else:
                class_instance.hardest_step_deltaG = input_graph.free_energy[vlabel_to_vind_dict[class_instance.hardest_step]]
                #graph.nodes[class_instance.hardest_step]["free_energy"]

        class_instance.path_dict = {"byproducts": class_instance.byproducts,
                                    "unsolved_prereqs": class_instance.unsolved_prereqs,
                                    "solved_prereqs": class_instance.solved_prereqs,
                                    "all_prereqs": class_instance.all_prereqs, "cost": class_instance.cost,
                                    "path": class_instance.path,
                                    "overall_free_energy_change": class_instance.overall_free_energy_change,
                                    "hardest_step": class_instance.hardest_step,
                                    "description": class_instance.description, "pure_cost": class_instance.pure_cost,
                                    "hardest_step_deltaG": class_instance.hardest_step_deltaG,
                                    "full_path": class_instance.full_path}

        return class_instance


Mapping_PR_Dict = Dict[int, Dict[int, ReactionPath]]


class ReactionNetwork(MSONable):
    """
       Class to build a reaction network from entries

    """

    def __init__(self, input_entries: List[MoleculeEntry], electron_free_energy=-2.15):
        """
        Initilize the ReacitonNetwork object attributes
        :param input_entries: [MoleculeEntry]: List of MoleculeEntry objects
        :param electron_free_energy: The Gibbs free energy of an electron. Defaults to -2.15 eV, the value at which the LiEC SEI forms

        """

        #self.graph = nx.DiGraph()
        self.reachable_nodes = []
        self.unsolvable_PRs = []
        self.graph = GraphTool()
        self.reactions = []
        self.input_entries = input_entries
        self.entry_ids = {e.entry_id for e in self.input_entries}
        self.electron_free_energy = electron_free_energy
        self.entries = {}
        self.entries_list = []
        self.num_starts = 0
        self.weight = None
        self.PR_record = None
        self.Reactant_record = None
        self.min_cost = {}
        self.not_reachable_nodes = []

        print(len(self.input_entries), "input entries")


        connected_entries = []
        for entry in self.input_entries:
            if len(entry.molecule) > 1:
                if nx.is_weakly_connected(entry.graph):
                    connected_entries.append(entry)
            else:
                connected_entries.append(entry)
        print(len(connected_entries), "connected entries")

        get_formula = lambda x: x.formula
        get_Nbonds = lambda x: x.Nbonds
        get_charge = lambda x: x.charge
        get_free_energy = lambda x: x.free_energy

        sorted_entries_0 = sorted(connected_entries, key=get_formula)
        for k1, g1 in itertools.groupby(sorted_entries_0, get_formula):
            sorted_entries_1 = sorted(list(g1), key=get_Nbonds)
            self.entries[k1] = {}

            for k2, g2 in itertools.groupby(sorted_entries_1, get_Nbonds):
                sorted_entries_2 = sorted(list(g2), key=get_charge)
                self.entries[k1][k2] = {}
                for k3, g3 in itertools.groupby(sorted_entries_2, get_charge):
                    sorted_entries_3 = list(g3)
                    sorted_entries_3.sort(key=get_free_energy)
                    if len(sorted_entries_3) > 1:
                        unique = []
                        for entry in sorted_entries_3:
                            isomorphic_found = False
                            for ii, Uentry in enumerate(unique):
                                if entry.mol_graph.isomorphic_to(Uentry.mol_graph):
                                    isomorphic_found = True
                                    # print("Isomorphic entries with equal charges found!")
                                    if entry.free_energy != None and Uentry.free_energy != None:
                                        if entry.free_energy < Uentry.free_energy:
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
            # if "ind" in entry.parameters.keys():
            #     pass
            # else:
            entry.parameters["ind"] = ii

        self.entries_list = sorted(self.entries_list, key=lambda x: x.parameters["ind"])

    @staticmethod
    def softplus(free_energy: float) -> float:
        """
            Method to determine edge weight using softplus cost function
        :param free_energy: float
        :return: float
        """
        return float(np.log(1 + (273.0 / 500.0) * np.exp(free_energy)))

    @staticmethod
    def exponent(free_energy: float) -> float:
        """
            Method to determine edge weight using exponent cost function
        :param free_energy: float
        :return: float
        """
        return float(np.exp(free_energy))

    def build(self, reaction_types={"RedoxReaction","IntramolSingleBondChangeReaction", "IntermolecularReaction",
                                    "CoordinationBondChangeReaction"}) -> nx.DiGraph:
        """
            A method to build the reaction network graph
        :param reaction_types: set of all the reactions class to include while building the graph
        :return: nx.DiGraph
        """

        #self.graph.add_nodes_from(range(len(self.entries_list)), bipartite=0)
        self.graph.graph.add_vertex(len(self.entries_list))
        self.graph.bipartite.a = [0]*len(self.entries_list)
        for v in self.graph.graph.vertices():
            self.graph.node_label[v] = str(int(v))
        self.graph.vind_to_vlabel_dict = {int(n): self.graph.node_label[n] for n in self.graph.graph.vertices()}
        self.graph.vlabel_to_vind_dict = {self.graph.node_label[n]: int(n) for n in self.graph.graph.vertices()}


        # node_label_dict = {n: self.graph.node_label[n] for n in self.graph.graph.vertices()}
        # print("@@", node_label_dict)

        reaction_types = [load_class(str(self.__module__) + "." + s) for s in reaction_types]
        for r in reaction_types:
            if r.__name__ == "ConcertedReaction":
                self.reactions = self.reactions+[r.generate(self.entries_list)]
            else:
                self.reactions = self.reactions+[r.generate(self.entries)]
        self.reactions = [i for i in self.reactions if i]
        self.reactions = list(itertools.chain.from_iterable(self.reactions))
        redox_c = 0
        inter_c = 0
        intra_c = 0
        coord_c = 0
        for r in self.reactions:
            if r.reaction_type()["class"] == "RedoxReaction":
                redox_c = redox_c + 1
                r.electron_free_energy = self.electron_free_energy
            elif r.reaction_type()["class"] == "IntramolSingleBondChangeReaction":
                intra_c = intra_c+1
            elif r.reaction_type()["class"] == "IntermolecularReaction":
                inter_c = inter_c+1
            elif r.reaction_type()["class"] == "CoordinationBondChangeReaction":
                coord_c = coord_c+1
            r.graph_representation(self.graph)
            #self.add_reaction(r.graph_representation(self.graph))
        print("redox: ", redox_c, "inter: ", inter_c, "intra: ", intra_c, "coord: ", coord_c)
        self.PR_record = self.build_PR_record()
        self.Reactant_record = self.build_reactant_record()
        print(self.graph.graph.num_vertices(ignore_filter=True))
        print(self.graph.graph.num_edges(ignore_filter=True))
        for v in self.graph.graph.vertices():
            self.graph.vfilter[v] = True
        for e in self.graph.graph.edges():
            self.graph.efilter[e] = True
        return self.graph

    def add_reaction(self, graph_representation: nx.DiGraph):
        """
            A method to add a single reaction to the ReactionNetwork.graph attribute
        :param graph_representation: Graph representation of a reaction, obtained from ReactionClass.graph_representation
        """
        self.graph.graph.add_vertex(list(graph_representation.graph.vertices())[0])
        #self.graph.add_nodes_from(graph_representation.nodes(data=True))
        #self.graph.add_edges_from(graph_representation.edges(data=True))

    def build_PR_record(self) -> Mapping_Record_Dict:
        """
            A method to determine all the reaction nodes that have a the same PR in the ReactionNetwork.graph
        :return: a dict of the form {int(node1): [all the reaction nodes with PR of node1, ex "2+PR_node1, 3"]}
        """

        PR_record = {}
        indices_0 = [i for i, x in enumerate(self.graph.bipartite.a) if x == 0]
        indices_1 = [i for i, x in enumerate(self.graph.bipartite.a) if x == 1]

        for i in indices_0:
            PR_record[i] = []

        for i in indices_1:
            if "+PR" in self.graph.node_label[i]:
                PR = int(self.graph.node_label[i].split(",")[0].split("+PR_")[1])
                PR_record[PR].append(self.graph.node_label[i])

        return PR_record

    def build_reactant_record(self) -> Mapping_Record_Dict:
        """
            A method to determine all the reaction nodes that have the same non PR reactant node in the ReactionNetwork.graph
        :return: a dict of the form {int(node1): [all the reaction nodes with non PR reactant of node1, ex "node1+PR_2, 3"]}
        """
        Reactant_record = {}
        indices_0 = [i for i, x in enumerate(self.graph.bipartite.a) if x == 0]
        indices_1 = [i for i, x in enumerate(self.graph.bipartite.a) if x == 1]

        for i in indices_0:
            Reactant_record[i] = []

        for i in indices_1:
            if "+PR" in self.graph.node_label[i]:
                non_PR_reactant = int(self.graph.node_label[i].split(",")[0].split("+PR_")[0])
                Reactant_record[non_PR_reactant].append(self.graph.node_label[i])

        # for node in self.graph.nodes():
        #     if self.graph.nodes[node]["bipartite"] == 0:
        #         Reactant_record[node] = []
        # for node in self.graph.nodes():
        #     if self.graph.nodes[node]["bipartite"] == 1:
        #         non_PR_reactant = node.split(",")[0].split("+PR_")[0]
        #         Reactant_record[int(non_PR_reactant)].append(node)
        return Reactant_record

    def solve_prerequisites(self, starts: List[int], weight: str, max_iter=20, save=False,
                            filename=None):  # -> Tuple[Union[Dict[Union[int, Any], dict], Any], Any]:
        """
            A method to solve the all the prerequisites found in ReactionNetwork.graph. By solving all PRs, it gives
            information on whether 1. if a path exist from any of the starts to all other molecule nodes, 2. if so what
            is the min cost to reach that node from any of the start, 3. if there is no path from any of the starts to a
            any of the molecule node, 4. for molecule nodes where the path exist, characterize the in the form of ReactionPath
        :param starts: List(molecular nodes), list of molecular nodes of type int found in the ReactionNetwork.graph
        :param target: a single molecular node of type int found in the ReactionNetwork.graph
        :param weight: "softplus" or "exponent", type of cost function to use when calculating edge weights
        :param max_iter: maximum number of iterations to try to solve all the PRs
        :return: PRs: PR_paths: dict that defines a path from each node to a start,
                of the form {int(node1): {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}}, int(node2):...}
        :return: min_cost: dict with minimum cost from path start to a node, of from {node: float},
                if no path exist, value is "no_path", if path is unsolved yet, value is "unsolved_path"
        :return: graph: ReactionNetwork.graph of type nx.DiGraph with updated edge weights based on solved PRs
        """

        t1 = time.time()
        PRs = {}
        old_solved_PRs = []
        new_solved_PRs = ["placeholder"]
        old_attrs = {}
        new_attrs = {}
        #self.weight = weight
        self.weight = eval('self.graph.'+weight)
        self.num_starts = len(starts)
        self.PR_byproducts = {}

        if self.graph.graph.num_vertices(ignore_filter=True) == 0:
            self.build()
        # v = graph_tool.util.find_vertex(self.graph.graph, self.graph.node_label, '127+PR_0,10')
        # print(v)
        # eg = self.graph.graph.edge(127, int(v[0]))
        # print(eg)
        # print("ooooo",self.graph.softplus[eg])
        orig_graph = copy.deepcopy(self.graph)
        for start in starts:
            PRs[start] = {}

        for PR in PRs:
            for start in starts:
                if start == PR:
                    PRs[PR][start] = ReactionPath.characterize_path([start], weight, self.min_cost, self.graph)
                else:
                    PRs[PR][start] = ReactionPath(None)
            old_solved_PRs.append(PR)
            self.min_cost[PR] = PRs[PR][PR].cost

        indices_0 = [i for i, x in enumerate(self.graph.bipartite.a) if x == 0]
        for i in indices_0:
            if i not in PRs:
                PRs[i] = {}

        ii = 0
        while (len(new_solved_PRs) > 0 or old_attrs != new_attrs) and ii < max_iter:
            min_cost = {}
            cost_from_start = {}
            for PR in PRs:
                cost_from_start[PR] = {}
                self.PR_byproducts[PR] = {}
                min_cost[PR] = 10000000000000000.0
                for start in PRs[PR]:
                    if PRs[PR][start].path == None:
                        cost_from_start[PR][start] = "no_path"
                    else:
                        cost_from_start[PR][start] = PRs[PR][start].cost
                        if PRs[PR][start].cost < min_cost[PR]:
                            min_cost[PR] = PRs[PR][start].cost
                            self.PR_byproducts[PR]["byproducts"] = PRs[PR][start].byproducts
                            self.PR_byproducts[PR]["start"] = start
                for start in starts:
                    if start not in cost_from_start[PR]:
                        cost_from_start[PR][start] = "unsolved"

            PRs, cost_from_start, min_cost = self.find_path_cost(starts, self.weight, old_solved_PRs,
                                                                 cost_from_start, min_cost, PRs)
            solved_PRs = copy.deepcopy(old_solved_PRs)
            solved_PRs, new_solved_PRs, cost_from_start = self.identify_solved_PRs(PRs, solved_PRs, cost_from_start)

            print(ii, len(old_solved_PRs), len(new_solved_PRs))

            attrs = self.update_edge_weights(min_cost, orig_graph, weight)

            self.min_cost = copy.deepcopy(min_cost)
            old_solved_PRs = copy.deepcopy(solved_PRs)
            old_attrs = copy.deepcopy(new_attrs)
            new_attrs = copy.deepcopy(attrs)

            ii += 1
        #dumpfn(PRs, "finalPRcheck_PRs_IN_TEST.json", default=lambda o: o.as_dict)
        print(time.time()-t1)
        self.solved_PRs = copy.deepcopy(old_solved_PRs)

        self.final_PR_check(PRs, weight)
        if save:
            if filename is None:
                print("Provide filename to save the PRs, for now saving as PRs.json")
                filename = "PRs.json"
            dumpfn(PRs, filename, default=lambda o: o.as_dict)
        print('not reachable nodes:', self.not_reachable_nodes)
        return PRs

    def parse_path(self, path):
        nodes = []
        PR = []
        Reactants = []
        for step in path:
            if isinstance(step, int):
                nodes.append(step)
            elif "PR_" in step:
                nodes = nodes + [step.split("+")[0]]
                Reactants.append(int(step.split("+")[0]))
                PR.append(int(step.split("+")[1].split("PR_")[1].split(",")[0]))
                nodes = nodes + step.split("+")[1].split("PR_")[1].split(",")
            else:
                assert (("," in step), True)
                nodes = nodes + step.split(",")
        nodes.pop(0)
        if len(nodes) != 0:
            nodes.pop(-1)
        return nodes, PR, Reactants

    def find_path_cost(self, starts, weight, old_solved_PRs, cost_from_start, min_cost, PRs):
        """
            A method to characterize the path to all the PRs. Characterize by determining if the path exist or not, and
            if so, is it a minimum cost path, and if so set PRs[node][start] = ReactionPath(path)
        :param starts: List(molecular nodes), list of molecular nodes of type int found in the ReactionNetwork.graph
        :param target: a single molecular node of type int found in the ReactionNetwork.graph
        :param weight: "softplus" or "exponent", type of cost function to use when calculating edge weights
        :param old_solved_PRs: list of PRs (molecular nodes of type int) that are already solved
        :param cost_from_start: dict of type {node1: {start1: float, start2: float}, node2: {...}}
        :param min_cost: dict with minimum cost from path start to a node, of from {node: float},
                if no path exist, value is "no_path", if path is unsolved yet, value is "unsolved_path"
        :param PRs: dict that defines a path from each node to a start, of the form {int(node1):
                {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}}, int(node2):...}
        :return: PRs: updated PRs based on new PRs solved
        :return: cost_from_start: updated cost_from_start based on new PRs solved
        :return: min_cost: updated min_cost based on new PRs solved
        """

        not_reachable_nodes_for_start = {}
        #self.weight = eval('self.graph.' + weight)
        wrong_paths = {}
        dist_and_path = {}
        self.num_starts = len(starts)
        for start in starts:
            not_reachable_nodes_for_start[start] = []
            dist_map, pred_map = graph_tool.search.dijkstra_search(self.graph.graph, self.weight, source=start)

            paths = {}
            for v in self.graph.graph.vertices():
                if self.graph.bipartite[v] == 0:
                    #paths[int(v)] = []
                    if dist_map.a[int(v)] != np.inf:
                        stop = True
                        # paths[int(v)]["cost"] = dist_map.a[int(v)]
                        # paths[int(v)]["path"] = []
                        path_map = []
                        istart = start
                        path_map.append(int(v))
                        p = int(pred_map.a[int(v)])
                        while stop:
                            if v != istart:
                                path_map.append(p)
                                if p == istart:
                                    stop = False
                                else:
                                    p = int(pred_map.a[int(p)])
                            else:
                                stop = False
                    else:
                        path_map = "no_path"
                    if isinstance(path_map, list):
                        path_map.reverse()
                        path = []
                        for node in path_map:
                            path.append(self.graph.vind_to_vlabel_dict[node])
                        paths[int(v)] = path
                    else:
                        paths[int(v)] = path_map

            # print(dist_map)
            # print(paths)
            # dist, paths = nx.algorithms.shortest_paths.weighted.single_source_dijkstra(self.graph, start,weight=self.weight)
            dist_and_path[start] = {}
            wrong_paths[start] = []
            for node in range(len(self.entries_list)):
                if node not in paths.keys():
                    not_reachable_nodes_for_start[start].append(int(node))
            for node in paths:
                if self.graph.bipartite[node] == 0:
                    if node not in self.reachable_nodes:
                        self.reachable_nodes.append(int(node))

                    dist_and_path[start][int(node)] = {}
                    dist_and_path[start][node]["cost"] = dist_map[node]
                    dist_and_path[start][node]["path"] = paths[node]
                    nodes = []
                    PR = []
                    Reactants = []
                    for step in paths[node]:
                        if isinstance(step, int):
                            nodes.append(step)
                        elif "PR_" in step:
                            nodes = nodes + [step.split("+")[0]]
                            Reactants.append(int(step.split("+")[0]))
                            PR.append(int(step.split("+")[1].split("PR_")[1].split(",")[0]))
                            if node in PR:
                                if node not in wrong_paths[start]:
                                    wrong_paths[start].append(int(node))
                            nodes = nodes + step.split("+")[1].split("PR_")[1].split(",")
                        else:
                            assert (("," in step), True)
                            nodes = nodes + step.split(",")
                    nodes.pop(0)
                    if len(nodes) != 0:
                        nodes.pop(-1)
                    dist_and_path[start][node]["all_nodes"] = nodes
                    dist_and_path[start][node]["PRs"] = PR
                    dist_and_path[start][node]["reactant"] = Reactants

        for node in self.graph.graph.vertices():
            if self.graph.bipartite[node] == 0:
                if node not in self.reachable_nodes:
                    self.not_reachable_nodes.append(node)

        for start in not_reachable_nodes_for_start:
            for node in not_reachable_nodes_for_start[start]:
                PRs[node][start] = ReactionPath(None)
                cost_from_start[node][start] = "no_path"
        #print(wrong_paths)
        fixed_paths = {}
        for start in wrong_paths:
            fixed_paths[start] = {}
            for node in wrong_paths[start]:
                fixed_paths[start][node] = {}
                gv = self.graph.ignorenodes(self.find_or_remove_bad_nodes([node]) + self.not_reachable_nodes)
                #print(gv.num_vertices())
                v_map, e_map = graph_tool.topology.shortest_path(gv,
                                                         self.graph.graph.vertex(start),
                                                         self.graph.graph.vertex(node), weights= self.weight)

                    # length, dij_path = nx.algorithms.simple_paths._bidirectional_dijkstra(
                    #     self.graph,
                    #     source=hash(start),
                    #     target=hash(node),
                    #     ignore_nodes=self.find_or_remove_bad_nodes([node] + self.not_reachable_nodes),
                    #     weight=self.weight)

                if v_map == []:
                    fixed_paths[start][node]["cost"] = "no_cost"
                    fixed_paths[start][node]["path"] = "no_path"
                else:
                    path = []
                    length = 0
                    for v in v_map:
                        path.append(self.graph.vind_to_vlabel_dict[v])
                    for e in e_map:
                        length = length + self.weight[e]
                    fixed_paths[start][node]["cost"] = length
                    fixed_paths[start][node]["path"] = path

        for start in starts:
            for node in fixed_paths[start]:
                if fixed_paths[start][node]["path"] == "no_path":
                    dist_and_path[start][node] = {}
                    pass
                else:
                    dist_and_path[start][node]["cost"] = fixed_paths[start][node]["cost"]
                    dist_and_path[start][node]["path"] = fixed_paths[start][node]["path"]
                    nodes, PR, reactant = self.parse_path(dist_and_path[start][node]["path"])
                    dist_and_path[start][node]["all_nodes"] = nodes
                    dist_and_path[start][node]["PRs"] = PR
                    dist_and_path[start][node]["reactant"] = reactant
            dist_and_path[start] = {key: value for key, value in
                                    sorted(dist_and_path[start].items(), key=lambda item: int(item[0]))}
        for start in starts:
            for node in dist_and_path[start]:
                if node not in old_solved_PRs:
                    if dist_and_path[start][node] == {}:
                        PRs[node][start] = ReactionPath(None)
                        cost_from_start[node][start] = "no_path"
                    elif dist_and_path[start][node]["cost"] >= 10000000000000000.0:
                        PRs[node][start] = ReactionPath(None)

                    else:
                        path_class = ReactionPath.characterize_path(dist_and_path[start][node]["path"], self.weight,
                                                                    self.min_cost, self.graph,
                                                                    old_solved_PRs, self.PR_byproducts, PRs)
                        cost_from_start[node][start] = path_class.cost

                        if len(path_class.unsolved_prereqs) == 0:
                            PRs[node][start] = path_class
                        if path_class.cost < min_cost[node]:
                            min_cost[node] = path_class.cost
                            self.PR_byproducts[node]["byproducts"] = path_class.byproducts
                            self.PR_byproducts[node]["start"] = start



        # # print("@@@", PRs)
        # #print("!!",min_cost)
        # self.num_starts = len(starts)
        # self.weight = eval('self.graph.' + weight)
        # for PR in PRs:
        #     reachable = False
        #     if all(start in PRs[PR].keys() for start in starts):
        #         for start in starts:
        #             if PRs[PR][start].path is not None:
        #                 reachable = True
        #     else:
        #         reachable = True
        #     if not reachable:
        #         if PR not in self.not_reachable_nodes:
        #             self.not_reachable_nodes.append(PR)
        #
        # indices_0 = [i for i, x in enumerate(self.graph.bipartite.a) if x == 0]
        # for i in indices_0:
        #     if i in old_solved_PRs:
        #         indices_0.remove(i)
        #
        #
        # # for node in self.graph.nodes():
        # #     if self.graph.nodes[node]["bipartite"] == 0 and node not in old_solved_PRs:# and node != target:
        # for vertex in indices_0:
        #     for start in starts:
        #         if start not in PRs[vertex]:
        #             path_exists = True
        #             gv = self.graph.ignorenodes(self.find_or_remove_bad_nodes([vertex]))
        #             vlist, elist = graph_tool.topology.shortest_path(gv, start, vertex, self.weight, False)
        #             if elist == []:
        #                 PRs[vertex][start] = ReactionPath(None)
        #                 path_exists = False
        #                 cost_from_start[vertex][start] = "no_path"
        #             if path_exists:
        #                 path_class = ReactionPath.characterize_path(vlist, elist, weight, self.min_cost, self.graph,
        #                                                             old_solved_PRs)
        #                 cost_from_start[vertex][start] = path_class.cost
        #
        #                 if len(path_class.unsolved_prereqs) == 0:
        #                     PRs[vertex][start] = path_class
        #                 if path_class.cost < min_cost[vertex]:
        #                     min_cost[vertex] = path_class.cost
        #                 # if vertex == 9:
        #                 #     print("999", vertex, path_class.path, path_class.cost, path_class.unsolved_prereqs,  min_cost[vertex])
        #                 #     print(len(path_class.unsolved_prereqs))


        return PRs, cost_from_start, min_cost

    def identify_solved_PRs(self, PRs, solved_PRs, cost_from_start):
        """
            A method to identify new solved PRs after each iteration
        :param PRs: dict that defines a path from each node to a start, of the form {int(node1):
                {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}}, int(node2):...}
        :param solved_PRs: list of PRs (molecular nodes of type int) that are already solved
        :param cost_from_start: dict of type {node1: {start1: float, start2: float}, node2: {...}}
        :return: solved_PRs: list of all the PRs(molecular nodes of type int) that are already solved plus new PRs solved in the current iteration
        :return: new_solved_PRs: list of just the new PRs(molecular nodes of type int) solved during current iteration
        :return: cost_from_start: updated dict of cost_from_start based on the new PRs solved during current iteration
        """
        new_solved_PRs = []
        for PR in PRs:
            if PR not in solved_PRs:
                if len(PRs[PR].keys()) == self.num_starts:
                    solved_PRs.append(PR)
                    new_solved_PRs.append(PR)
                else:
                    best_start_so_far = [None, 10000000000000000.0]
                    for start in PRs[PR]:
                        if PRs[PR][start] is not None:  # ALWAYS TRUE shoudl be != {}
                            # if PRs[PR][start] == "unsolved": #### DOES THIS EVER HAPPEN ---- NEED TO FIX
                            #     print("ERROR: unsolved should never be encountered here!")
                            if PRs[PR][start].cost < best_start_so_far[1]:
                                best_start_so_far[0] = start
                                best_start_so_far[1] = PRs[PR][start].cost
                    if best_start_so_far[0] is not None:
                        num_beaten = 0
                        for start in cost_from_start[PR]:
                            if start != best_start_so_far[0]:
                                if cost_from_start[PR][start] == "no_path":
                                    num_beaten += 1
                                elif cost_from_start[PR][start] > best_start_so_far[1]:
                                    num_beaten += 1
                        if num_beaten == self.num_starts - 1:
                            solved_PRs.append(PR)
                            new_solved_PRs.append(PR)

        return solved_PRs, new_solved_PRs, cost_from_start

    def update_edge_weights(self, min_cost: Dict[int, float], orig_graph: GraphTool, weight) -> Dict[Tuple[int, str], Dict[
        str, float]]:  # , solved_PRs: List[int], new_attrs:Dict[Tuple[int, str],Dict[str,float]]):
        """
            A method to update the ReactionNetwork.graph edge weights based on the new cost of solving PRs
        :param min_cost: dict with minimum cost from path start to a node, of from {node: float},
                if no path exist, value is "no_path", if path is unsolved yet, value is "unsolved_path"
        :param orig_graph: ReactionNetwork.graph of type nx.Digraph before the start of current iteration of updates
        :return: attrs: dict of form {(node1, node2), {"softplus": float, "exponent": float, "weight: 1}, (node2, node3): {...}}
                dict of all the edges to update the weights of
        """
        if self.graph.graph.num_vertices(ignore_filter=True) == 0:
            self.build()

        if self.PR_record is None:
            self.PR_record = self.build_PR_record()

       # print("***", orig_graph.graph.edge(20, 1))
        vlabel_to_vind_dict = {orig_graph.node_label[n]: int(n) for n in orig_graph.graph.vertices()}
        vind_to_vlabel_dict = {int(n): orig_graph.node_label[n] for n in orig_graph.graph.vertices()}
        #vlabel_to_vind_dict = {self.graph.node_label[n]: int(n) for n in self.graph.graph.vertices()}
        #vind_to_vlabel_dict = {int(n): self.graph.node_label[n] for n in self.graph.graph.vertices()}
        attrs = {}
        og_graph_weight = eval('orig_graph.' + weight)
        self_graph_weight = eval('self.graph.' + weight)
        for PR_ind in min_cost:
            for rxn_node in self.PR_record[PR_ind]:
                #print(orig_graph.graph, orig_graph.node_label, str(rxn_node))
                rxn_node_vertex = vlabel_to_vind_dict[str(rxn_node)]
                 #   graph_tool.util.find_vertex(orig_graph.graph, orig_graph.node_label, str(rxn_node))
                non_PR_reactant_node = int(rxn_node.split(",")[0].split("+PR_")[0])
                og_edge = orig_graph.graph.edge(int(non_PR_reactant_node), int(rxn_node_vertex))
                #self_edge = self.graph.graph.edge(int(non_PR_reactant_node), int(rxn_node_vertex))
                #if vind_to_vlabel_dict[int(og_edge.source())] == "127" and vind_to_vlabel_dict[int(og_edge.target())] == '127+PR_0,10':
                    #print("##", orig_graph.softplus[og_edge])
                    #print(og_graph_weight[og_edge], min_cost[PR_ind])

                #og_graph_weight[og_edge] = og_graph_weight[og_edge] + min_cost[PR_ind]
                attrs[(vind_to_vlabel_dict[int(og_edge.source())], vind_to_vlabel_dict[int(og_edge.target())])] = {weight: og_graph_weight[og_edge] + min_cost[PR_ind]}
                self_graph_weight[self.graph.graph.edge(int(og_edge.source()), int(og_edge.target()))] = og_graph_weight[og_edge] + min_cost[PR_ind]
        # for edge in orig_graph.graph.edges():
        #     #print("&&",edge,og_graph_weight[edge], og_graph_weight[orig_graph.graph.edge(int(edge.source()), int(edge.target()))], type(int(edge.target())), type(edge.target()))
        #     self_graph_weight[self.graph.graph.edge(int(edge.source()), int(edge.target()))] = og_graph_weight[edge]

        # 0.007207114516399871
        # 1.063579623578771
        # 1.0707867380951708
        #
        # 0.007207114516399871
        # 1.063579623578771
        # 1.0707867380952154}
        #self.weight = og_graph_weight
        #print("***", orig_graph.graph.edge(20, 1))
        #print("***", self.graph.graph.edge(20, 1))
        #print("***", self.weight)
        # attrs = {}
        # for PR_ind in min_cost:
        #     for rxn_node in self.PR_record[PR_ind]:
        #         non_PR_reactant_node = int(rxn_node.split(",")[0].split("+PR_")[0])
        #         attrs[(non_PR_reactant_node, rxn_node)] = {
        #             self.weight: orig_graph[non_PR_reactant_node][rxn_node][self.weight] + min_cost[PR_ind]}
        # nx.set_edge_attributes(self.graph, attrs)
        #print(attrs)
        return attrs

    def final_PR_check(self, PRs: Mapping_PR_Dict, weight):
        """
            A method to check errors in the path attributes of the PRs with a path, if no path then prints no path from any start to a given
        :param PRs: dict that defines a path from each node to a start, of the form {int(node1):
                {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}}, int(node2):...}
        """
        vlabel_to_vind_dict = {self.graph.node_label[n]: int(n) for n in self.graph.graph.vertices()}
        vind_to_vlabel_dict = {int(n): self.graph.node_label[n] for n in self.graph.graph.vertices()}
        print(vlabel_to_vind_dict)
        print(vind_to_vlabel_dict)
        for PR in PRs:
            path_found = False
            if PRs[PR] != {}:
                for start in PRs[PR]:
                    if PRs[PR][start].path != None:
                        path_found = True
                        #print(PRs[PR][start].path)
                        # i = 0
                        # vlist = []
                        # elist = []
                        # for n in PRs[PR][start].path:
                        #     vlist.append(vlabel_to_vind_dict[str(n)])
                        # while i < len(vlist)-1:
                        #     elist.append(self.graph.graph.edge(int(vlist[i]), int(vlist[i+1])))
                        #     i=i+1
                        path_dict_class = ReactionPath.characterize_path_final(PRs[PR][start].path, weight,
                                                                               self.min_cost, self.graph, self.solved_PRs,
                                                                               PR_byproduct_dict = self.PR_byproducts, PR_paths = PRs)
                        if abs(path_dict_class.cost - path_dict_class.pure_cost) > 0.0001:
                            print("WARNING: cost mismatch for PR", PR, path_dict_class.cost, path_dict_class.pure_cost,
                                  path_dict_class.full_path)
                if not path_found:
                    print("No path found from any start to PR", PR)
            else:
                print("Unsolvable path from any start to PR", PR)

    def find_or_remove_bad_nodes(self, nodes: List[str], remove_nodes=False) -> List[str] or nx.DiGraph:
        """
            A method to either create a list of the nodes a path solving method should ignore or generate a graph without
            all the nodes it a path solving method should not use in obtaining a path.
        :param nodes: List(molecular nodes), list of molecular nodes of type int found in the ReactionNetwork.graph
        that should be ignored when solving a path
        :param remove_nodes: if False (default), return list of bad nodes, if True, return a version of
        ReactionNetwork.graph (of type nx.Digraph) from with list of bad nodes are removed
        :return: if remove_nodes = False -> list[node], if remove_nodes = True -> nx.DiGraph
        """
        if self.graph.graph.num_vertices(ignore_filter=True) == 0:
            self.graph = self.build()
        if self.PR_record is None:
            self.PR_record = self.build_PR_record()
        if self.Reactant_record is None:
            self.Reactant_record = self.build_reactant_record()
        bad_nodes = []
        for node in nodes:
            for bad_node in self.PR_record[node]:
                bad_nodes.append(bad_node)
            for bad_nodes2 in self.Reactant_record[node]:
                bad_nodes.append(bad_nodes2)
        if remove_nodes:
            pruned_graph = copy.deepcopy(self.graph)
            pruned_graph.remove_nodes_from(bad_nodes)
            return pruned_graph
        else:
            return bad_nodes




    def valid_shortest_simple_paths(self, start: int, target: int, PRs=[]):  # -> Generator[List[str]]:????
        """
            A method to determine shortest path from start to target
        :param start: molecular node of type int from ReactionNetwork.graph
        :param target: molecular node of type int from ReactionNetwork.graph
        :param PRs: not used currently?
        :return: nx.path_generator of type generator
        """

        gv = self.graph.ignorenodes(self.find_or_remove_bad_nodes([target]))
        # bad_nodes = PRs
        # bad_nodes.append(target)
        # valid_graph = self.find_or_remove_bad_nodes(bad_nodes, remove_nodes=True)
        # vlist, elist = graph_tool.topology.shortest_path(gv, start, target, self.weight, False)
        # return vlist, elist
        paths = graph_tool.topology.all_shortest_paths(gv, start, target, weights=self.weight)
        #paths = graph_tool.topology.all_paths(gv, start, target)
        #return nx.shortest_simple_paths(gv, hash(start), hash(target), weight=self.weight)
        return paths

    def find_paths(self, starts, target, weight, num_paths=10, solved_PRs_path=None, ignorenode=[]):  # -> ??
        """
            A method to find the shorted parth from given starts to a target
        :param starts: starts: List(molecular nodes), list of molecular nodes of type int found in the ReactionNetwork.graph
        :param target: a single molecular node of type int found in the ReactionNetwork.graph
        :param weight: "softplus" or "exponent", type of cost function to use when calculating edge weights
        :param num_paths: Number (of type int) of paths to find. Defaults to 10.
        :param solved_PRs_path: dict that defines a path from each node to a start,
                of the form {int(node1): {int(start1}: {ReactionPath object}, int(start2): {ReactionPath object}}, int(node2):...}
                if None, method will solve PRs
        :param solved_min_cost: dict with minimum cost from path start to a node, of from {node: float},
                if no path exist, value is "no_path", if path is unsolved yet, value is "unsolved_path",
                of None, method will solve for min_cost
        :param updated_graph: nx.DiGraph with udpated edge weights based on the solved PRs, if none, method will solve for PRs and update graph accordingly
        :param save: if True method will save PRs paths, min cost and updated graph after all the PRs are solved,
                    if False, method will not save anything (default)
        :return: PR_paths: solved dict of PRs
        :return: paths: list of paths (number of paths based on the value of num_paths)
        """

        self.weight = eval('self.graph.'+weight)
        self.num_starts = len(starts)
        paths = []
        c = itertools.count()
        my_heapq = []
        if self.graph.graph.num_vertices(ignore_filter=True) == 0:
            self.build()
        print("Solving prerequisites...")
        if solved_PRs_path is None:
            self.min_cost = {}
            #self.graph = self.build()
            PR_paths = self.solve_prerequisites(starts, weight)

        else:
            PR_paths = {}
            for key in solved_PRs_path:
                PR_paths[int(key)] = {}
                for start in solved_PRs_path[key]:
                    PR_paths[int(key)][int(start)] = copy.deepcopy(solved_PRs_path[key][start])

            for key in PR_paths:
                self.min_cost[int(key)] = 10000000000000000.0
                for start in PR_paths[key]:
                    if self.min_cost[int(key)] == 10000000000000000.0:
                        self.min_cost[int(key)] = PR_paths[key][start].cost
                    elif self.min_cost[int(key)] > PR_paths[key][start].cost:
                        self.min_cost[int(key)] = PR_paths[key][start].cost
            self.build()
            self.build_PR_record()
            self.weight = weight
            for PR in self.PR_record:
                for rxn_node in self.PR_record[PR]:
                    non_PR_reactant_node = int(rxn_node.split(",")[0].split("+PR_")[0])
                    self.graph[non_PR_reactant_node][rxn_node][self.weight] = self.graph[non_PR_reactant_node][rxn_node][
                                                                              weight] + self.min_cost[PR]
        print("Finding paths...")
        vlabel_to_vind_dict = {self.graph.node_label[n]: int(n) for n in self.graph.graph.vertices()}
        vind_to_vlabel_dict = {int(n): self.graph.node_label[n] for n in self.graph.graph.vertices()}
        for start in starts:
            # ind = 0
            # p = []
            # for v in vlist:
            #     p.append(vind_to_vlabel_dict[v])
            # print(p)
            # vlist, elist = self.valid_shortest_simple_paths(start, target, ignorenode)
            # path_dict_class2 = ReactionPath.characterize_path_final(vlist, elist, self.weight, self.min_cost,
            #                                                         self.graph, PR_paths)
            # print(path_dict_class2.path_dict)

            paths = self.valid_shortest_simple_paths(start, target, ignorenode)

            for path in paths:
                # i = 0
                # vlist = []
                # elist = []
                # vlist = path
                # while i < len(path) - 1:
                #     elist.append(self.graph.graph.edge(int(vlist[i]), int(vlist[i + 1])))
                #     i = i +1
                # print(path)
                p = []
                for v in path:
                    p.append(self.graph.vind_to_vlabel_dict[v])
                path_dict_class2 = ReactionPath.characterize_path_final(p, weight, self.min_cost,self.graph, self.solved_PRs,
                                                                               PR_byproduct_dict = self.PR_byproducts, PR_paths = PR_paths)


                print(path_dict_class2.path_dict)
        #     for path in self.valid_shortest_simple_paths(start, target, ignorenode):
        #         # i = 0
        #         # vlist = []
        #         # elist = []
        #         # for n in path:
        #         #     vlist.append(vind_to_vlabel_dict[n])
        #         # while i < len(vlist) - 1:
        #         #     elist.append(self.graph.graph.edge(int(vlist[i]), int(vlist[i + 1])))
        #         if ind == num_paths:
        #             break
        #         else:
        #             ind += 1
        #             path_dict_class2 = ReactionPath.characterize_path_final(vlist, elist, self.weight, self.min_cost,
        #                                                                     self.graph, PR_paths)
        #             heapq.heappush(my_heapq, (path_dict_class2.cost, next(c), path_dict_class2))
        #
        # while len(paths) < num_paths and my_heapq:
        #     # Check if any byproduct could yield a prereq cheaper than from starting molecule(s)?
        #     (cost_HP, _x, path_dict_HP_class) = heapq.heappop(my_heapq)
        #     print(len(paths), cost_HP, len(my_heapq), path_dict_HP_class.path_dict)
        #     paths.append(
        #         path_dict_HP_class.path_dict)  ### ideally just append the class, but for now dict for easy printing
        #
        # #print(PR_paths)
        # print(paths)

        #return PR_paths, paths
