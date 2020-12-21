from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from atomate.qchem.database import QChemCalcDb
from pymatgen import Molecule
import copy
from itertools import combinations_with_replacement, combinations
from pymatgen.analysis.fragmenter import open_ring
from monty.serialization import dumpfn, loadfn
import numpy as np
import pathos
import gzip
from pulp import *

def convert_atomic_numbers_to_stoi_dict(atomic_numbers):
    '''

    :param atomic_numbers: a list of atomic numbers
    :return: {'Li':1, '110':0,'C':3,...} zero padding for non-existing elements
    '''
    atomic_num_to_element = {1:'H',3:'Li',6:'C',8:'O',9:'F',15:'P'}
    elements = ['H','Li','C','O','F','P']
    stoi_dict = {}

    for num in atomic_numbers:
        if atomic_num_to_element[num] in stoi_dict.keys():
            stoi_dict[atomic_num_to_element[num]] += 1
        else:
            stoi_dict[atomic_num_to_element[num]] = 1
    for ele in elements:
        if ele not in stoi_dict.keys():
            stoi_dict[ele] = 0
    return stoi_dict

def combine_stoi_dict(stoi_dict1, stoi_dict2):
    new_stoi_dict = {'C': stoi_dict1['C'] + stoi_dict2['C'], 'O': stoi_dict1['O'] + stoi_dict2['O'],
                     'H': stoi_dict1['H'] + stoi_dict2['H'], 'Li': stoi_dict1['Li'] + stoi_dict2['Li'],
                     'P': stoi_dict1['P'] + stoi_dict2['P'], 'F':stoi_dict1['F'] + stoi_dict2['F']}
    return new_stoi_dict

def combine_stoi_dict_triple(stoi_dict1, stoi_dict2, stoi_dict3):
    new_stoi_dict = {'C': stoi_dict1['C'] + stoi_dict2['C'] + stoi_dict3['C'], 'O': stoi_dict1['O'] + stoi_dict2['O'] + stoi_dict3['O'],
                     'H': stoi_dict1['H'] + stoi_dict2['H'] + stoi_dict3['H'], 'Li': stoi_dict1['Li'] + stoi_dict2['Li'] + stoi_dict3['Li'],
                     'P': stoi_dict1['P'] + stoi_dict2['P'] + stoi_dict3['P'], 'F':stoi_dict1['F'] + stoi_dict2['F'] + stoi_dict3['F']}
    return new_stoi_dict


def identify_same_stoi_mol_pairs(mol_graphs):
    '''
    :param mol_graphs: A list of mol_graphs
    :return: A dictionary with all mol pairs(or single molecule) that adds up to the same stoichiometry
    '''
    stoi_list = []
    final_dict = {}
    num_mols = len(mol_graphs)
    all_mol_pair_index = list(combinations_with_replacement(range(num_mols), 2))
    print('Number of all molecule pairs:', len(all_mol_pair_index), flush=True)
    for i, mol_pair in enumerate(all_mol_pair_index):
        print('mol_pair index:',i, flush=True)
        index1 = mol_pair[0]
        index2 = mol_pair[1]
        pair_key = str(index1) + '_' + str(index2)
        mol1 = mol_graphs[index1].molecule
        mol2 = mol_graphs[index2].molecule
        stoi_dict1 = convert_atomic_numbers_to_stoi_dict(mol1.atomic_numbers)
        stoi_dict2 = convert_atomic_numbers_to_stoi_dict(mol2.atomic_numbers)
        stoi_dict = combine_stoi_dict(stoi_dict1, stoi_dict2)
        if stoi_dict in stoi_list:
            index_in_list = stoi_list.index(stoi_dict)
            final_dict[index_in_list].append(pair_key)
        else:
            final_dict[len(stoi_list)] = [pair_key]
            stoi_list.append(stoi_dict)
    for i, mol_graph in enumerate(mol_graphs):
        mol = mol_graph.molecule
        stoi_dict = convert_atomic_numbers_to_stoi_dict(mol.atomic_numbers)
        if stoi_dict in stoi_list:
            index_in_list = stoi_list.index(stoi_dict)
            final_dict[index_in_list].append(str(i))
        else:
            final_dict[len(stoi_list)] = [str(i)]
            stoi_list.append(stoi_dict)

    return stoi_list, final_dict

def identify_same_stoi_mol_pairs_ABC_DE(mol_graphs):
    '''
    For identifying A + B + C -> D + E reaction candidates that match the stoichiometry.
    TODO: Only allowing triples to include H/Li/F single atom for now.
    :param mol_graphs: A list of mol_graphs
    :return: A dictionary with all mol pairs(or single molecule) that adds up to the same stoichiometry
    '''
    stoi_list = []
    final_dict = {}
    num_mols = len(mol_graphs)
    all_mol_triple_index = list(combinations_with_replacement(range(num_mols), 3))
    all_mol_pair_index = list(combinations_with_replacement(range(num_mols), 2))

    for mol_pair in all_mol_pair_index:
        index1 = mol_pair[0]
        index2 = mol_pair[1]
        pair_key = str(index1) + '_' + str(index2)
        mol1 = mol_graphs[index1].molecule
        mol2 = mol_graphs[index2].molecule
        stoi_dict1 = convert_atomic_numbers_to_stoi_dict(mol1.atomic_numbers)
        stoi_dict2 = convert_atomic_numbers_to_stoi_dict(mol2.atomic_numbers)
        stoi_dict = combine_stoi_dict(stoi_dict1, stoi_dict2)
        if stoi_dict in stoi_list:
            index_in_list = stoi_list.index(stoi_dict)
            final_dict[index_in_list].append(pair_key)
        else:
            final_dict[len(stoi_list)] = [pair_key]
            stoi_list.append(stoi_dict)

    allowed_symbol_list = ['Li', 'H', 'F']
    for mol_triple in all_mol_triple_index:
        index1 = mol_triple[0]
        index2 = mol_triple[1]
        index3 = mol_triple[2]

        triple_key = str(index1) + '_' + str(index2) + '_' + str(index3)
        mol1 = mol_graphs[index1].molecule
        mol2 = mol_graphs[index2].molecule
        mol3 = mol_graphs[index3].molecule
        if (len(mol1) == 1 and mol1.sites[0].specie.symbol in allowed_symbol_list) or \
                (len(mol2) == 1 and mol2.sites[0].specie.symbol in allowed_symbol_list) or \
                (len(mol3) == 1 and mol3.sites[0].specie.symbol in allowed_symbol_list):

            stoi_dict1 = convert_atomic_numbers_to_stoi_dict(mol1.atomic_numbers)
            stoi_dict2 = convert_atomic_numbers_to_stoi_dict(mol2.atomic_numbers)
            stoi_dict3 = convert_atomic_numbers_to_stoi_dict(mol3.atomic_numbers)
            stoi_dict = combine_stoi_dict_triple(stoi_dict1, stoi_dict2, stoi_dict3)
            if stoi_dict in stoi_list:
                index_in_list = stoi_list.index(stoi_dict)
                final_dict[index_in_list].append(triple_key)
            else:
                final_dict[len(stoi_list)] = [triple_key]
                stoi_list.append(stoi_dict)

    for i, mol_graph in enumerate(mol_graphs):
        mol = mol_graph.molecule
        stoi_dict = convert_atomic_numbers_to_stoi_dict(mol.atomic_numbers)
        if stoi_dict in stoi_list:
            index_in_list = stoi_list.index(stoi_dict)
            final_dict[index_in_list].append(str(i))
        else:
            final_dict[len(stoi_list)] = [str(i)]
            stoi_list.append(stoi_dict)

    return stoi_list, final_dict

def identify_concerted_reaction(mol_graphs1, mol_graphs2, allowed_bond_change=4):
    '''
    break A, B once each. Not considering breaking two or more bonds in a mol.
    :param mol_graph1:
    :param mol_graph2:
    :return:
    '''
    num_atoms1 = sum(len(mol_graph) for mol_graph in mol_graphs1)
    num_atoms2 = sum(len(mol_graph) for mol_graph in mol_graphs2)
    assert num_atoms1 == num_atoms2
    is_concerted_reaction = False
    reactant_edges = []
    product_edges = []
    TR = []
    TP = []
    reactant_idx = 0
    product_idx = 0
    for i, mol_graph in enumerate(mol_graphs1):
        reactant_edges += [(item[0]+reactant_idx, item[1]+reactant_idx) for item in mol_graph.graph.edges]
        TR += list(mol_graph.molecule.atomic_numbers)
        reactant_idx += len(mol_graph)
    for i, mol_graph in enumerate(mol_graphs2):
        product_edges += [(item[0]+product_idx,item[1]+product_idx) for item in mol_graph.graph.edges]
        TP += list(mol_graph.molecule.atomic_numbers)
        product_idx += len(mol_graph)
    if abs(len(reactant_edges) - len(product_edges)) > allowed_bond_change:
        return is_concerted_reaction

    opt_model = LpProblem(name="MIP Model")

    A = set(range(num_atoms1))

    alpha_vars = {(i, j, k, l): LpVariable(cat=LpBinary, name="alpha_{0}_{1}_{2}_{3}".format(i, j, k, l))
                  for (i, j) in reactant_edges for (k, l) in product_edges}

    y_vars = {(i, k): LpVariable(cat=LpBinary, name="y_{0}_{1}".format(i, k)) for i in A for k in A}

    # constraints
    # Constraint 2 requires that each atom in the reactants maps to exactly one atom in the products. Constraint 3 requires that each atom in the products maps to exactly one atom in the reactants.
    for i in A:
        opt_model += lpSum([y_vars[i, k] for k in A]) == 1
    for k in A:
        opt_model += lpSum([y_vars[i, k] for i in A]) == 1

    # Constraint 4 allows only atoms of the same type to map to one another.
    for i in A:
        for k in A:
            if TR[i] != TP[k]:
                opt_model += y_vars[i, k] == 0

    # Constraints 5 and 6 deﬁne each αijkl variable, permitting it to take the value of one only if the reactant bond (i,j) maps to the product bond (k,l).
    for (i, j) in reactant_edges:
        for (k, l) in product_edges:
            opt_model += alpha_vars[i, j, k, l] <= y_vars[i, k] + y_vars[i, l]
            opt_model += alpha_vars[i, j, k, l] <= y_vars[j, k] + y_vars[j, l]

    objective = lpSum(1 - lpSum(alpha_vars[i, j, k, l] for (k, l) in product_edges) for (i, j) in reactant_edges) + \
                lpSum(1 - lpSum(alpha_vars[i, j, k, l] for (i, j) in reactant_edges) for (k, l) in product_edges)

    opt_model.sense = LpMinimize
    opt_model.setObjective(objective)
    opt_model.solve()
    print("Production Costs = ", value(opt_model.objective), flush=True)
    if value(opt_model.objective) != None and value(opt_model.objective) <= allowed_bond_change:
        is_concerted_reaction = True

    return is_concerted_reaction


class FindConcertedReactions:
    def __init__(self, entries_list, name):
        """
        class for finding concerted reactions
        Args:
        :param entries_list, entries_list = [MoleculeEntry]
        :param name: name for saving various dicts.
        """
        self.entries_list = entries_list
        self.name = name

        return

    def find_concerted_candidates(self):
        '''
        Find concerted reaction candidates by finding reactant-product pairs that match the stoichiometry.
        Args:
        :param entries: ReactionNetwork(input_entries).entries_list, entries_list = [MoleculeEntry]
        :param name: name for saving self.unique_mol_graph_dict.
        :return: self.concerted_rxns_to_determine: [['15_43', '19_43']]: [[str(reactants),str(products)]]
                 reactants and products are separated by "_".
                 The number correspond to the index of a mol_graph in self.unique_mol_graphs_new.
        '''
        print("Finding concerted reaction candidates!", flush=True)
        self.unique_mol_graphs_new = []
        # For duplicate mol graphs, create a map between later species with former ones
        # Only determine once for each unique mol_graph.
        self.unique_mol_graph_dict = {}

        for i in range(len(self.entries_list)):
            print('mol_graph index:',i, flush=True)
            mol_graph = self.entries_list[i].mol_graph
            found = False
            for j in range(len(self.unique_mol_graphs_new)):
                new_mol_graph = self.unique_mol_graphs_new[j]
                if mol_graph.molecule.composition.alphabetical_formula == new_mol_graph.molecule.composition.alphabetical_formula and mol_graph.isomorphic_to(
                        new_mol_graph):
                    found = True
                    self.unique_mol_graph_dict[i] = j
                    continue
            if not found:
                self.unique_mol_graph_dict[i] = len(self.unique_mol_graphs_new)
                self.unique_mol_graphs_new.append(mol_graph)
        # dumpfn(self.unique_mol_graph_dict, self.name + "_unique_mol_graph_map.json")
        # find all molecule pairs that satisfy the stoichiometry constraint
        stoi_list, species_same_stoi_dict = identify_same_stoi_mol_pairs(self.unique_mol_graphs_new)

        cnt = 0
        number_concerted_reactions = 0
        self.concerted_rxns_to_determine = []
        print('Number of keys in species_same_stoi_dict:', len(species_same_stoi_dict), flush=True)
        number_elements_dict = {key:len(species_same_stoi_dict[key]) for key in species_same_stoi_dict}
        dumpfn(number_elements_dict, 'number_elements_dict.json')
        for i, key in enumerate(species_same_stoi_dict.keys()):
            print('key_index:',i, flush=True)
            species_list = species_same_stoi_dict[key]
            print('species list length:', len(species_list), flush=True)
            if species_list != []:
                for j in range(len(species_list)):
                    reac = species_list[j]
                    for k in range(j+1, len(species_list)):
                        prod = species_list[k]

                        split_reac = reac.split('_')
                        split_prod = prod.split('_')
                        length_reac = len(split_reac)
                        for i in range(length_reac):
                            if len(split_reac) > i:
                                item = split_reac[i]
                                if item in split_prod:
                                    prod_index = split_prod.index(item)
                                    split_reac.pop(i)
                                    split_prod.pop(prod_index)

                        # split_reac_unique = [x for x in split_reac if x not in split_prod]
                        # split_prod_unique = [x for x in split_prod if x not in split_reac]
                        if len(split_reac) != 0 and len(split_prod) != 0:
                            reac_unique = '_'.join(split_reac)
                            prod_unique = '_'.join(split_prod)
                            #if [reac_unique, prod_unique] not in self.concerted_rxns_to_determine:
                            self.concerted_rxns_to_determine.append([reac_unique, prod_unique])
                            number = len(self.concerted_rxns_to_determine)
                            if number > 100000:
                                dumpfn(self.concerted_rxns_to_determine, 'concerted_candidates_{}.json'.format(cnt))
                                self.concerted_rxns_to_determine = []
                                number_concerted_reactions += number
                                cnt += 1
        dumpfn(self.concerted_rxns_to_determine, 'concerted_candidates_{}.json'.format(cnt))
        number = len(self.concerted_rxns_to_determine)
        number_concerted_reactions += number


        print('number of concerted candidates:', number_concerted_reactions, flush=True)
        #dumpfn(self.concerted_rxns_to_determine, 'concerted_candidates.json')

        return

    def find_concerted_candidates_ABC_DE(self):
        '''
        Find concerted reaction candidates by finding reactant-product pairs that match the stoichiometry,
        for some A + B + C -> D + E reactions with single Li/H/F in A/B/C.
        This adds on to the previous function.
        Args:
        :param entries: ReactionNetwork(input_entries).entries_list, entries_list = [MoleculeEntry]
        :param name: name for saving self.unique_mol_graph_dict.
        :return: self.concerted_rxns_to_determine: [['15_43', '19_43']]: [[str(reactants),str(products)]]
                 reactants and products are separated by "_".
                 The number correspond to the index of a mol_graph in self.unique_mol_graphs_new.
        '''
        print("Finding concerted reaction candidates!", flush=True)
        self.unique_mol_graphs_new = []
        # For duplicate mol graphs, create a map between later species with former ones
        # Only determine once for each unique mol_graph.
        self.unique_mol_graph_dict = {}

        for i in range(len(self.entries_list)):
            mol_graph = self.entries_list[i].mol_graph
            found = False
            for j in range(len(self.unique_mol_graphs_new)):
                new_mol_graph = self.unique_mol_graphs_new[j]
                if mol_graph.molecule.composition.alphabetical_formula == new_mol_graph.molecule.composition.alphabetical_formula and mol_graph.isomorphic_to(
                        new_mol_graph):
                    found = True
                    self.unique_mol_graph_dict[i] = j
                    continue
            if not found:
                self.unique_mol_graph_dict[i] = len(self.unique_mol_graphs_new)
                self.unique_mol_graphs_new.append(mol_graph)
        # dumpfn(self.unique_mol_graph_dict, self.name + "_unique_mol_graph_map.json")
        # find all molecule pairs that satisfy the stoichiometry constraint
        stoi_list, species_same_stoi_dict = identify_same_stoi_mol_pairs_ABC_DE(self.unique_mol_graphs_new)

        self.concerted_rxns_to_determine = []
        for i, key in enumerate(species_same_stoi_dict.keys()):
            species_list = species_same_stoi_dict[key]
            new_species_list_reactant = []
            new_species_list_product = []
            for species in species_list:
                new_species_list_reactant.append(species)
                new_species_list_product.append(species)
            if new_species_list_reactant != [] and new_species_list_product != []:
                reactants = new_species_list_reactant
                products = new_species_list_product
                for j in range(len(reactants)):
                    reac = reactants[j]
                    for k in range(len(products)):
                        prod = products[k]
                        if k <= j:
                            continue
                        else:
                            split_reac = reac.split('_')
                            split_prod = prod.split('_')
                            if (len(split_reac) == 2 and len(split_prod) == 3) or (
                                    len(split_reac) == 3 and len(split_prod) == 2):
                                if len(split_reac) == 2 and len(split_prod) == 3:
                                    split_reac, split_prod = split_prod, split_reac
                                if (split_prod[0] in split_reac) or (split_prod[1] in split_reac):
                                    continue
                                else:
                                    if [reac, prod] not in self.concerted_rxns_to_determine:
                                        self.concerted_rxns_to_determine.append([reac, prod])

        print('number of ABC_DE concerted candidates:', len(self.concerted_rxns_to_determine), flush=True)
        dumpfn(self.concerted_rxns_to_determine, 'concerted_candidates_ABC_DE.json')

        return

    def find_concerted_reactions(self, args):
        '''
        Determine whether one reaction in self.concerted_rxns_to_determine is concerted reaction < n bond change.
        Note that if a reaction is elementary (in class "RedoxReaction", "IntramolSingleBondChangeReaction", "IntermolecularReaction",
        "CoordinationBondChangeReaction"), it is also considered concerted. It has to be removed later on in the ReactionNetwork class.
        Note: if it's single reactant and single product, only consider concerted rxns with <=2 bond change, regardless of the user defined allowed number of bond change.
        :param index: Index in self.concerted_rxns_to_determine
        :return: valid_reactions:[['15_43', '19_43']]: [[str(reactants),str(products)]]
                 reactants and products are separated by "_".
                 The number correspond to the index of a mol_graph in self.unique_mol_graphs_new.
        '''
        index, restart, allowed_bond_change = args[0], args[1], args[2]
        print('current index:', index, flush=True)
        valid_reactions = []

        reac = self.concerted_rxns_to_determine[index][0]
        prod = self.concerted_rxns_to_determine[index][1]
        split_reac = reac.split('_')
        split_prod = prod.split('_')
        print('reactant:', reac, flush=True)
        print('product:', prod, flush=True)

        if restart and [reac, prod] in self.loaded_valid_reactions:
            valid_reactions.append([reac, prod])
            print('found!', flush=True)
            return valid_reactions
        if restart and [reac, prod] in self.loaded_invalid_reactions:
            print('found!', flush=True)
            return valid_reactions

        mol_graphs1 = [self.unique_mol_graphs_new[int(i)] for i in split_reac]
        mol_graphs2 = [self.unique_mol_graphs_new[int(i)] for i in split_prod]

        found = False

        if len(mol_graphs1) == 1 and len(mol_graphs2) == 1:
            if identify_concerted_reaction(mol_graphs1, mol_graphs2, 2):
                if [reac, prod] not in valid_reactions:
                    found = True
                    valid_reactions.append([reac, prod])
        else:
            if identify_concerted_reaction(mol_graphs1, mol_graphs2, allowed_bond_change):
                if [reac, prod] not in valid_reactions:
                    found = True
                    valid_reactions.append([reac, prod])

        output = "valid_reactions_bond_change_{}".format(allowed_bond_change)
        output_not_concerted = "invalid_reactions_bond_change_{}".format(allowed_bond_change)
        if found:
            with open(output, 'a+') as f:
                rxn = [reac, prod]
                f.write(str(rxn) + '\n')
        elif not found:
            with open(output_not_concerted, 'a+') as f:
                rxn = [reac, prod]
                f.write(str(rxn) + '\n')
        return valid_reactions

    def find_concerted_multiprocess(self, num_processors, allowed_bond_change=4, restart=False):
        '''
        Use multiprocessing to determine concerted reactions in parallel.
        Args:
        :param num_processors:
        :param reaction_type: Can choose from "break2_form2" and "break1_form1"
        :return: self.valid_reactions:[['15_43', '19_43']]: [[str(reactants),str(products)]]
                 reactants and products are separated by "_".
                 The number correspond to the index of a mol_graph in self.unique_mol_graphs_new.
        '''
        if restart:
            self.loaded_valid_reactions = []
            self.loaded_invalid_reactions = []
            valid_reactions_to_load = "valid_reactions_bond_change_{}".format(allowed_bond_change)
            invalid_reactions_to_load = "invalid_reactions_bond_change_{}".format(allowed_bond_change)
            f = open(valid_reactions_to_load, "r")
            contents = f.readlines()
            f.close()
            self.loaded_valid_reactions = []
            for i, content in enumerate(contents):
                new_content = content.replace('\n', '').replace("'", "").strip('][').split(', ')
                self.loaded_valid_reactions.append(new_content)
                if new_content in self.concerted_rxns_to_determine:
                    self.concerted_rxns_to_determine.remove(new_content)

            f = open(invalid_reactions_to_load, "r")
            contents = f.readlines()
            f.close()
            for i, content in enumerate(contents):
                new_content = content.replace('\n', '').replace("'", "").strip('][').split(', ')
                self.loaded_invalid_reactions.append(new_content)
                if new_content in self.concerted_rxns_to_determine:
                    self.concerted_rxns_to_determine.remove(new_content)
        #print("Remaining number of concerted reactions to determine:", len(self.concerted_rxns_to_determine),flush=True)
        print("Finding concerted reactions, allowing {} bond changes!".format(allowed_bond_change), flush=True)

        from pathos.multiprocessing import ProcessingPool as Pool
        self.valid_reactions = []
        cnt = 0
        for file in os.listdir('.'):
            if file.startswith('concerted_candidates') and file.endswith('.json'):
                print('current file:', file, flush=True)
                print('{}th json file!'.format(cnt), flush=True)
                cnt += 1
                self.concerted_rxns_to_determine = loadfn(file)
                nums = list(np.arange(len(self.concerted_rxns_to_determine)))
                args = [(i, restart, allowed_bond_change) for i in nums]
                pool = Pool(num_processors)
                results = pool.map(self.find_concerted_reactions, args)
                for i in range(len(results)):
                    valid_reactions = results[i]
                    self.valid_reactions += valid_reactions
                if restart:
                    self.valid_reactions += self.loaded_valid_reactions
        # dumpfn(self.valid_reactions, name + "_valid_concerted_rxns.json")
        return

    def get_final_concerted_reactions(self, name, num_processors, allowed_bond_change=4, restart=False):
        '''
        This is for getting the final set of concerted reactions: entry index corresponds to the index in self.entries_list.
        Args:
        :param name: name for saving self.valid_reactions. self.valid_reactions has the following form:
               [["0_1", "6_46"]]: [[str(reactants), str(products)]] reactants and products are separated by "_".
               The number correspond to the index of a mol_graph in self.unique_mol_graphs_new.
        :param num_processors:
        :param reaction_type: Can choose from "break2_form2" and "break1_form1"

        :return: [['15_43', '19_43']]: [[str(reactants),str(products)]]
                 reactants and products are separated by "_".
                 The number correspond to the index of a mol_graph in self.entries_list.
        '''

        self.find_concerted_candidates()
        self.find_concerted_multiprocess(num_processors, allowed_bond_change, restart=restart)
        print("Summarizing concerted reactions!", flush=True)
        self.final_concerted_reactions = []

        for i in range(len(self.valid_reactions)):
            rxn = self.valid_reactions[i]
            reactant_nodes = rxn[0].split('_')
            product_nodes = rxn[1].split('_')
            reactant_candidates = []
            product_candidates = []
            for reac in reactant_nodes:
                reac_cands = []
                for map_key in self.unique_mol_graph_dict.keys():
                    if self.unique_mol_graph_dict[map_key] == int(reac):
                        reac_cands.append(map_key)
                reactant_candidates.append(reac_cands)
            for prod in product_nodes:
                prod_cands = []
                for map_key in self.unique_mol_graph_dict.keys():
                    if self.unique_mol_graph_dict[map_key] == int(prod):
                        prod_cands.append(map_key)
                product_candidates.append(prod_cands)
            print('reactant candidates:', reactant_candidates, flush=True)
            print('product candidates:', product_candidates, flush=True)

            if len(reactant_candidates) == 1 and len(product_candidates) == 1:
                for j in reactant_candidates[0]:
                    for k in product_candidates[0]:
                        self.final_concerted_reactions.append([str(j), str(k)])

            if len(reactant_candidates) == 2 and len(product_candidates) == 1:
                for j in reactant_candidates[0]:
                    for k in reactant_candidates[1]:
                        for m in product_candidates[0]:
                            if int(j) <= int(k):
                                reactant_name = str(j) + '_' + str(k)
                            else:
                                reactant_name = str(k) + '_' + str(j)
                            self.final_concerted_reactions.append([reactant_name, str(m)])

            if len(reactant_candidates) == 1 and len(product_candidates) == 2:
                for j in reactant_candidates[0]:
                    for m in product_candidates[0]:
                        for n in product_candidates[1]:
                            if int(m) <= int(n):
                                product_name = str(m) + '_' + str(n)
                            else:
                                product_name = str(n) + '_' + str(m)
                            self.final_concerted_reactions.append([str(j), product_name])

            if len(reactant_candidates) == 2 and len(product_candidates) == 2:
                for j in reactant_candidates[0]:
                    for k in reactant_candidates[1]:
                        for m in product_candidates[0]:
                            for n in product_candidates[1]:
                                if int(j) <= int(k):
                                    reactant_name = str(j) + '_' + str(k)
                                else:
                                    reactant_name = str(k) + '_' + str(j)
                                if int(m) <= int(n):
                                    product_name = str(m) + '_' + str(n)
                                else:
                                    product_name = str(n) + '_' + str(m)
                                self.final_concerted_reactions.append([reactant_name, product_name])

            if len(reactant_candidates) == 2 and len(product_candidates) == 3:
                reactant_candidates, product_candidates = product_candidates, reactant_candidates
            if len(reactant_candidates) == 3 and len(product_candidates) == 2:
                for j in reactant_candidates[0]:
                    for k in reactant_candidates[1]:
                        for p in reactant_candidates[2]:
                            for m in product_candidates[0]:
                                for n in product_candidates[1]:
                                    reactant_index_list = [int(j), int(k), int(p)]
                                    reactant_index_list.sort()
                                    reactant_name = str(reactant_index_list[0]) + '_' + str(
                                        reactant_index_list[1]) + '_' + str(reactant_index_list[2])
                                    if int(m) <= int(n):
                                        product_name = str(m) + '_' + str(n)
                                    else:
                                        product_name = str(n) + '_' + str(m)
                                    self.final_concerted_reactions.append([reactant_name, product_name])
        dumpfn(self.final_concerted_reactions, name + '_concerted_rxns_bond_change_{}.json'.format(allowed_bond_change))
        return self.final_concerted_reactions