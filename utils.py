import string
import gudhi as gd
import numpy as np

np.random.seed(0)


def find_cliques(edges, max_sz=None):
    """
    Generate a simplicial complex from the set of edges
    :param edges: Set of edges in the graph
    :param max_sz: Max size of a simplex
    :return: The simplicial complex
    """
    make_strings = isinstance(next(iter(edges)), str)
    edges = {frozenset(edge) for edge in edges}
    vertices = {vertex for edge in edges for vertex in edge}
    neighbors = {
        vtx: frozenset(({vtx} ^ e).pop() for e in edges if vtx in e) for vtx in vertices
    }
    if max_sz is None:
        max_sz = len(vertices)

    simplices = [set(), vertices, edges]
    shared_neighbors = {frozenset({vtx}): nb for vtx, nb in neighbors.items()}
    for j in range(2, max_sz):
        nxt_deg = set()
        for smplx in simplices[-1]:
            # split off random vertex
            rem = set(smplx)
            rv = rem.pop()
            rem = frozenset(rem)
            # find shared neighbors
            shrd_nb = shared_neighbors[rem] & neighbors[rv]
            shared_neighbors[smplx] = shrd_nb
            # and build containing simplices
            nxt_deg.update(smplx | {vtx} for vtx in shrd_nb)
        if not nxt_deg:
            break
        simplices.append(nxt_deg)
    if make_strings:
        for j in range(2, len(simplices)):
            simplices[j] = {*map("".join, map(sorted, simplices[j]))}
    return simplices


def gershgorin(matrix):
    """
    Outputs the estimated maximum eigenvalue of input matrix using Gershgorin circle theorem
    :param matrix: Input matrix for eigenvalue estimation
    :return: Estimate of maximum eigenvalue
    """
    ends = []
    for i, row in enumerate(matrix):
        ci = row[i]
        ri = np.sum(np.absolute(np.delete(row, i)))
        ends.append(ci + ri)
    return max(ends)


def make_simplicies(vertices_list, edge_length, max_d):
    characters = string.digits + string.ascii_letters

    skeletons_2d = [
        gd.RipsComplex(points=x, max_edge_length=edge_length) for x in vertices_list
    ]
    data_2d_simplex_tree = [
        skeleton.create_simplex_tree(max_dimension=max_d) for skeleton in skeletons_2d
    ]
    rips_lists = [
        list(simplex_tree.get_filtration()) for simplex_tree in data_2d_simplex_tree
    ]

    scs = []
    for rips_list in rips_lists:
        sc = []
        for simplex in rips_list:
            temp = ""
            for vertex in simplex[0]:
                temp = temp + characters[vertex]
            sc.append(temp)
        scs.append(sc)

    return scs
