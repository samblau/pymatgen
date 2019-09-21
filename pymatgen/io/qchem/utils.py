import re
import numpy as np
from scipy.optimize import leastsq
from collections import defaultdict
import itertools
from difflib import SequenceMatcher
from statistics import mean
import copy

import networkx as nx
import networkx.algorithms.isomorphism as iso

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph


def read_pattern(text_str, patterns, terminate_on_match=False,
                 postprocess=str):
    """
        General pattern reading on an input string

        Args:
            text_str (str): the input string to search for patterns
            patterns (dict): A dict of patterns, e.g.,
                {"energy": r"energy\\(sigma->0\\)\\s+=\\s+([\\d\\-.]+)"}.
            terminate_on_match (bool): Whether to terminate when there is at
                least one match in each key in pattern.
            postprocess (callable): A post processing function to convert all
                matches. Defaults to str, i.e., no change.

        Renders accessible:
            Any attribute in patterns. For example,
            {"energy": r"energy\\(sigma->0\\)\\s+=\\s+([\\d\\-.]+)"} will set the
            value of matches["energy"] = [[-1234], [-3453], ...], to the
            results from regex and postprocess. Note that the returned values
            are lists of lists, because you can grep multiple items on one line.
    """

    compiled = {
        key: re.compile(pattern, re.MULTILINE | re.DOTALL)
        for key, pattern in patterns.items()
    }
    matches = defaultdict(list)
    for key, pattern in compiled.items():
        for match in pattern.finditer(text_str):
            matches[key].append([postprocess(i) for i in match.groups()])
            if terminate_on_match:
                break
    return matches


def read_table_pattern(text_str,
                       header_pattern,
                       row_pattern,
                       footer_pattern,
                       postprocess=str,
                       attribute_name=None,
                       last_one_only=False):
    """
    Parse table-like data. A table composes of three parts: header,
    main body, footer. All the data matches "row pattern" in the main body
    will be returned.

    Args:
        text_str (str): the input string to search for patterns
        header_pattern (str): The regular expression pattern matches the
            table header. This pattern should match all the text
            immediately before the main body of the table. For multiple
            sections table match the text until the section of
            interest. MULTILINE and DOTALL options are enforced, as a
            result, the "." meta-character will also match "\n" in this
            section.
        row_pattern (str): The regular expression matches a single line in
            the table. Capture interested field using regular expression
            groups.
        footer_pattern (str): The regular expression matches the end of the
            table. E.g. a long dash line.
        postprocess (callable): A post processing function to convert all
            matches. Defaults to str, i.e., no change.
        attribute_name (str): Name of this table. If present the parsed data
            will be attached to "data. e.g. self.data["efg"] = [...]
        last_one_only (bool): All the tables will be parsed, if this option
            is set to True, only the last table will be returned. The
            enclosing list will be removed. i.e. Only a single table will
            be returned. Default to be True.

    Returns:
        List of tables. 1) A table is a list of rows. 2) A row if either a list of
        attribute values in case the the capturing group is defined without name in
        row_pattern, or a dict in case that named capturing groups are defined by
        row_pattern.
    """

    table_pattern_text = header_pattern + \
        r"\s*(?P<table_body>(?:" + row_pattern + r")+)\s*" + footer_pattern
    table_pattern = re.compile(table_pattern_text, re.MULTILINE | re.DOTALL)
    rp = re.compile(row_pattern)
    data = dict()
    tables = list()
    for mt in table_pattern.finditer(text_str):
        table_body_text = mt.group("table_body")
        table_contents = []
        for ml in rp.finditer(table_body_text):
            d = ml.groupdict()
            if len(d) > 0:
                processed_line = {k: postprocess(v) for k, v in d.items()}
            else:
                processed_line = [postprocess(v) for v in ml.groups()]
            table_contents.append(processed_line)
        tables.append(table_contents)
    if last_one_only:
        retained_data = tables[-1]
    else:
        retained_data = tables
    if attribute_name is not None:
        data[attribute_name] = retained_data
        return data
    return retained_data


def read_table_pattern_with_useful_header_footer(text_str,
                                                 header_pattern,
                                                 row_pattern,
                                                 footer_pattern,
                                                 postprocess=str,
                                                 attribute_name=None,
                                                 last_one_only=False,
                                                 num_rows=None):
    """
    Parse table-like data, retaining information from the header and footer. A
    table composes of three parts: header, main body, footer. All data, matching
    the header patter, row pattern, and footer pattern, will be returned.

    Args:
        text_str (str): the input string to search for patterns
        header_pattern (str): The regular expression pattern matches the
            table header. This pattern should match all the text
            immediately before the main body of the table. For multiple
            sections table match the text until the section of
            interest. MULTILINE and DOTALL options are enforced, as a
            result, the "." meta-character will also match "\n" in this
            section.
        row_pattern (str): The regular expression matches a single line in
            the table. Capture interested field using regular expression
            groups.
        footer_pattern (str): The regular expression matches the end of the
            table. E.g. a long dash line.
        postprocess (callable): A post processing function to convert all
            matches. Defaults to str, i.e., no change.
        attribute_name (str): Name of this table. If present the parsed data
            will be attached to "data. e.g. self.data["efg"] = [...]
        last_one_only (bool): All the tables will be parsed, if this option
            is set to True, only the last table will be returned. The
            enclosing list will be removed. i.e. Only a single table will
            be returned. Default to be True.
        num_rows (int): If not None (default), only accept a certain number of
            rows.

    Returns:
        Dict of lists of tables. 1) A table is a list of rows. 2) A row if
        either a list of attribute values in case the the capturing group is
        defined without name in row_pattern, or a dict in case that named
        capturing groups are defined by row_pattern.
    """

    # Determine number of rows allowed/expected
    # Default is 1 - infinity
    # if num_rows is given, only accept a table with a particular number of rows
    if num_rows is None:
        num_str = r")+)\s*"
    else:
        num_str = r"){" + str(num_rows) + r"})\s*"

    table_pattern_text = r"\s*(?P<table_header>(?:" + header_pattern + \
                         r"))" + r"\s*(?P<table_body>(?:" + row_pattern + \
                         num_str + r"(?P<table_footer>(?:" + footer_pattern + \
                         r"))\s*"
    table_pattern = re.compile(table_pattern_text, re.MULTILINE | re.DOTALL)
    hp = re.compile(header_pattern)
    rp = re.compile(row_pattern)
    fp = re.compile(footer_pattern)

    data = dict()
    headers = list()
    bodies = list()
    footers = list()
    for mt in table_pattern.finditer(text_str):
        table_header_text = mt.group("table_header")
        table_body_text = mt.group("table_body")
        table_footer_text = mt.group("table_footer")
        table_contents = list()

        hmatch = hp.match(table_header_text)
        hd = hmatch.groupdict()
        if len(hd) > 0:
            processed_header = {k: postprocess(v) for k, v in hd.items()}
        else:
            processed_header = [postprocess(v) for v in hmatch.groups()]
        headers.append(processed_header)

        for ml in rp.finditer(table_body_text):
            d = ml.groupdict()
            if len(d) > 0:
                processed_line = {k: postprocess(v) for k, v in d.items()}
            else:
                processed_line = [postprocess(v) for v in ml.groups()]
            table_contents.append(processed_line)
        bodies.append(table_contents)

        fmatch = fp.match(table_footer_text)
        fd = fmatch.groupdict()
        if len(fd) > 0:
            processed_footer = {k: postprocess(v) for k, v in fd.items()}
        else:
            processed_footer = [postprocess(v) for v in fmatch.groups()]
        footers.append(processed_footer)

    if last_one_only:
        retained_data = {"header": headers[-1],
                         "body": bodies[-1],
                         "footer": footers[-1]}
    else:
        retained_data = {"header": headers,
                         "body": bodies,
                         "footer": footers}

    if attribute_name is not None:
        data[attribute_name] = retained_data
        return data
    return retained_data


def lower_and_check_unique(dict_to_check):
    """
    Takes a dictionary and makes all the keys lower case. Also replaces
    "jobtype" with "job_type" just so that key specifically can be called
    elsewhere without ambiguity. Finally, ensures that multiple identical
    keys, that differed only due to different capitalizations, are not
    present. If there are multiple equivalent keys, an Exception is raised.

    Args:
        dict_to_check (dict): The dictionary to check and standardize

    Returns:
        to_return (dict): An identical dictionary but with all keys made
            lower case and no identical keys present.
    """
    if dict_to_check is None:
        return None
    else:
        to_return = {}
        for key in dict_to_check:
            new_key = key.lower()
            if new_key == "jobtype":
                new_key = "job_type"
            if new_key in to_return:
                if to_return[key] != to_return[new_key]:
                    raise Exception(
                        "Multiple instances of key " + new_key + " found with different values! Exiting...")
            else:
                try:
                    to_return[new_key] = dict_to_check.get(key).lower()
                except AttributeError:
                    to_return[new_key] = dict_to_check.get(key)
        return to_return


def process_parsed_coords(coords):
    """
    Takes a set of parsed coordinates, which come as an array of strings,
    and returns a numpy array of floats.
    """
    geometry = np.zeros(shape=(len(coords), 3), dtype=float)
    for ii, entry in enumerate(coords):
        for jj in range(3):
            geometry[ii, jj] = float(entry[jj])
    return geometry


def map_atoms_reaction(reactants, product):
    """
    Create a mapping of atoms between a set of reactant Molecules and a product
    Molecule.

    :param reactants: list of MoleculeGraph objects representing the reaction
        reactants
    :param product: MoleculeGraph object representing the reaction product

    NOTE: This currently only works with one product

    :return: dict {product_atom_index: reactant_atom_index}
    """

    def get_ranked_atom_dists(mol):
        dist_matrix = mol.distance_matrix

        result = dict()
        for num, row in enumerate(dist_matrix):
            ranking = np.argsort(row)
            # The first member will always be the atom itself, which should be excluded
            result[num] = ranking[1:]
        return result

    # Next, try to construct isomorphisms between reactant and product
    nm = iso.categorical_node_match("specie", "ERROR")
    # Prefer undirected graphs
    rct_graphs = [rct.graph.to_undirected() for rct in reactants]
    pro_graph = product.graph.to_undirected()

    pro_dists = get_ranked_atom_dists(product.molecule)
    ranking_by_reactant = list()
    for e, rct_graph in enumerate(rct_graphs):

        meta_iso = {e: set() for e in range(len(product.molecule))}
        matcher = iso.GraphMatcher(pro_graph, rct_graph, node_match=nm)

        # Compile all isomorphisms
        isomorphisms = [i for i in matcher.subgraph_isomorphisms_iter()]
        for isomorphism in isomorphisms:
            for pro_node, rct_node in isomorphism.items():
                meta_iso[pro_node].add(rct_node)

        meta_iso = {k: v for (k, v) in meta_iso.items() if v != set()}

        # Determine which nodes need to be checked
        disputed_nodes = set()
        for pro_node, rct_nodes in meta_iso.items():
            if len(rct_nodes) > 1:
                disputed_nodes.add(pro_node)

        average_ratios = list()
        rct_dists = get_ranked_atom_dists(reactants[e].molecule)
        for isomorphism in isomorphisms:
            ratios = list()

            for node in disputed_nodes:
                if node in isomorphism:
                    rct_dist = rct_dists[isomorphism[node]]
                    pro_dist_old = pro_dists[node]

                    pro_dist = list()
                    for n in pro_dist_old:
                        if n in isomorphism:
                            pro_dist.append(isomorphism[n])

                    matcher = SequenceMatcher(None, rct_dist, pro_dist)
                    ratios.append(matcher.ratio())

            average_ratios.append(mean(ratios))

        # Rank isomorphisms by the average SequenceMatch ratio
        ranking_by_reactant.append([isom for _, isom in sorted(zip(average_ratios, isomorphisms),
                                                         key=lambda pair: pair[0])])

    combinations = list(itertools.product(*ranking_by_reactant))

    mapping = None
    for combination in combinations:
        index_total = 0
        this_mapping = dict()

        for part in combination:
            for key, value in part.items():
                this_mapping[key] = value + index_total
            index_total += len(part.keys())

        # If this is a valid mapping - all atoms accounted for
        if len(this_mapping) == len(product):
            mapping = this_mapping
            break

    return mapping


def orient_molecule(mol_1, mol_2):
    """
    Determine the translation vector that minimizes the distances between
    corresponding atoms in two (isomorphic) molecules.

    :param mol_1: MoleculeGraph of the molecule that is not to be translated
    :param mol_2: MoleculeGraph of the molecule that is to be translated

    :return: np.ndarray representing a translation vector for mol_2
    """

    def atom_dist(n, vec):
        copy_2 = copy.deepcopy(mol_2)
        # Get distance between atom n in mol_1 and mol_2 after transformation
        trans = vec[:3]
        rot = vec[3:]

        for index, vec in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
            copy_2.molecule.rotate_sites(theta=rot[index],
                                         axis=vec,
                                         anchor=copy_2.molecule.center_of_mass)

        coord_n = copy_2.molecule.cart_coords[n] + trans

        return np.linalg.norm(mol_1.molecule.cart_coords[n] - coord_n)

    def all_dists(vec):
        return np.array([atom_dist(n, vec) for n in range(len(mol_1))])

    if not mol_1.isomorphic_to(mol_2):
        raise ValueError("Function coorient should only be used on isomorphic "
                         "MoleculeGraphs!")

    return leastsq(all_dists, np.zeros(6))[0]


def generate_string_start(reactants, product, strategy, reorder=False,
                          extend_structure=False, map_atoms=True,
                          separation_dist=0.5):
    """
    For a reaction of type A + B <-> C, manipulate C in such a way as to provide
    a reasonable starting guess for string calculations (FSM/GSM).

    :param reactants: list of Molecule objects representing the reaction
        reactants
    :param product: Molecule object representing the reaction product
    :param strategy: local_env NearNeighbors object used to generate
        MoleculeGraphs
    :param reorder: parameter for local_env strategies
    :param extend_structure: parameter for local_env strategies
    :param map_atoms: if True (default), use map_atoms_reaction to ensure that
        the nodes in the reactant and product graphs are the same. If False,
        the user must ensure that the product molecule has its atoms in the
        same order as the reactants in order for this function to work properly.
    :param separation_dist: distance (in Angsrom) to move each reacting species.
        A value of 0 means that the reactant fragments will be in exactly the
        same position that they are in the product molecule.

    NOTE: This currently only works with one product.

    :return: dict(group: [Molecule]), where group is either "reactants" or
        "products"
    """

    rct_mgs = list()
    species = list()
    coords = list()
    charge = product.charge
    spin = product.spin_multiplicity
    distance = 0
    rcts = copy.deepcopy(reactants)
    for rct in rcts:
        rct.translate_sites(vector=np.array([distance, distance, distance]))
        for site in rct:
            species.append(site.specie)
            coords.append(site.coords)
        rct_mgs.append(MoleculeGraph.with_local_env_strategy(rct, strategy,
                                                             reorder=reorder,
                                                             extend_structure=extend_structure))
        distance += 5
    # print(rct_mgs)

    # generate composite Molecule and MoleculeGraph including all reactants
    all_rct = Molecule(species, coords, charge=charge, spin_multiplicity=spin)
    all_rct_mg = MoleculeGraph.with_local_env_strategy(all_rct, strategy,
                                                       reorder=reorder,
                                                       extend_structure=extend_structure)

    pro_mg = MoleculeGraph.with_local_env_strategy(product, strategy,
                                                   reorder=reorder,
                                                   extend_structure=extend_structure)

    # perform atom mapping, and reorder product accordingly
    if map_atoms:
        mapping = map_atoms_reaction(rct_mgs, pro_mg)

        if mapping is None:
            raise ValueError("Reactant atoms cannot be mapped to product molecules using existing methods. "
                             "Please map atoms by hand and set map_atoms=False to try again.")

        species = [None for _ in range(len(product))]
        coords = [None for _ in range(len(product))]
        for e, site in enumerate(product):
            species[mapping[e]] = site.species
            coords[mapping[e]] = site.coords
        new_pro = Molecule(species, coords, charge=charge,
                           spin_multiplicity=spin)
        pro_mg = MoleculeGraph.with_local_env_strategy(new_pro, strategy,
                                                       reorder=reorder,
                                                       extend_structure=extend_structure)

    # print(all_rct_mg)
    # print(pro_mg)
    # break bonds to get reactants from product
    diff_graph = nx.difference(pro_mg.graph, all_rct_mg.graph)
    # print(diff_graph.edges())
    for bond in diff_graph.edges():
        pro_mg.break_edge(bond[0], bond[1], allow_reverse=True)

    frags = pro_mg.get_disconnected_fragments()
    # print(frags)

    coms = dict()
    for e, frag in enumerate(frags):
        coms[e] = np.array(frag.molecule.center_of_mass)

    # determine vectors along which to move fragments
    vectors = dict()
    for i, com_i in coms.items():
        for j, com_j in coms.items():
            if i < j and (i, j) not in vectors and (j, i) not in vectors:
                # direction chosen so vector can be applied to first index
                vec = com_i - com_j
                norm = np.linalg.norm(vec)
                if norm == 0:
                    vectors[(i, j)] = vec
                else:
                    vectors[(i, j)] = vec / norm

    # map fragments to reactants, and translate reactants to fragment positions
    frag_rct_map = dict()
    for f, frag in enumerate(frags):
        for r, rct_mg in enumerate(rct_mgs):
            if frag.isomorphic_to(rct_mg) and r not in frag_rct_map.values():
                frag_rct_map[f] = r
                orient_vec = orient_molecule(frag, rct_mg)
                trans = orient_vec[0:3]
                rot = orient_vec[3:]
                for index, vec in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
                    rct_mg.molecule.rotate_sites(theta=rot[index],
                                                 axis=vec,
                                                 anchor=rct_mg.molecule.center_of_mass)
                rct_mg.molecule.translate_sites(vector=trans)
                rct_mg.set_node_attributes()
                break
    # print(frag_rct_map)

    # apply separation vectors to reactants
    # reactants are now in right place
    for indices, vector in vectors.items():
        vector *= separation_dist
        rct_mgs[indices[0]].molecule.translate_sites(vector=vector)
        rct_mgs[indices[0]].set_node_attributes()

    return {"reactants": [r.molecule for r in rct_mgs],
            "products": [pro_mg.molecule]}
