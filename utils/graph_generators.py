'''
Created on December 26, 2017

@author: optas
TODO: Merge/ re-factor (fundamentals/graph + graph_roles)
PRELIMINARY CODE - not used/debugged yet. 

'''

import numpy as np
import random

from scipy.sparse import coo_matrix


def adjacency_from_edges(edges, n_nodes, sparse=True, dtype=np.int32):
    source_e = np.array([i[0] for i in edges])
    target_e = np.array([i[1] for i in edges])
    vals = np.ones_like(source_e, dtype=dtype)
    if sparse:
        res = coo_matrix((vals, (source_e, target_e)), shape=(n_nodes, n_nodes), dtype=dtype)
    else:
        raise NotImplementedError()
    return res


def gnm_random_graph(n, m, seed=None, directed=False):
    """Returns a `G_{n,m}` random graph.

    In the `G_{n,m}` model, a graph is chosen uniformly at random from the set
    of all graphs with `n` nodes and `m` edges.

    This algorithm should be faster than :func:`dense_gnm_random_graph` for
    sparse graphs.

    Parameters
    ----------
    n : int
        The number of nodes.
    m : int
        The number of edges.
    seed : int, optional
        Seed for random number generator (default=None).
    directed : bool, optional (default=False)
        If True return a directed graph

    See also
    --------
    dense_gnm_random_graph

    """
    max_edges = n * (n - 1)
    if not directed:
        max_edges /= 2.0
    if m >= max_edges:
        raise ValueError('Too many edjes.')

    nlist = np.arange(n)
    edge_count = 0
    edges = set()
    while edge_count < m:
        # generate random edge u,v
        u = random.choice(nlist)
        v = random.choice(nlist)
        if u == v or (u, v) in edges:
            continue
        else:
            edges.add((u, v))
            if not directed:
                edges.add((v, u))
            edge_count += 1

    return adjacency_from_edges(edges, n)


def SBM_from_class_labels(vertex_labels, p_matrix):
    'stochastic block model'
    n_vertices = len(vertex_labels)
    adjacency = np.zeros(shape=(n_vertices, n_vertices), dtype=np.bool)
    for row, _row in enumerate(adjacency):
        for col, _col in enumerate(adjacency[row]):
            community_a = vertex_labels[row]
            community_b = vertex_labels[col]
            p = random.random()
            val = p_matrix[community_a][community_b]

            if p <= val:
                adjacency[row][col] = 1
                adjacency[col][row] = 1
    return adjacency