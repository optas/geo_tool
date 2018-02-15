'''
Created on June 14, 2016

@author:    Panos Achlioptas
@contact:   pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

import numpy as np
import warnings
from collections import defaultdict

from general_tools.arrays.basics import unique_rows

from .. utils import linalg_utils as linalg_utils


def filter_vertices(self, keep_list):
    '''Filters the mesh to contain only the vertices in the input ``keep_list``.
    Also, it discards any triangles that do not contain vertices that all belong in the list.
    All changes happen in-place.
    TODO:  speed_up
    '''

    if len(keep_list) == 0:
        raise ValueError('Provided list of nodes is empty.')
    keep_list = np.unique(keep_list)
    delete_index = set(np.arange(self.num_vertices)) - set(keep_list)
    new_order = linalg_utils.order_of_elements_after_deletion(self.num_vertices, list(delete_index))
    self.vertices = self.vertices[keep_list, :]
    clean_triangles = np.array([set(tr) - delete_index == set(tr) for tr in self.triangles.tolist()])
    clean_triangles = self.triangles[clean_triangles]
    for x in np.nditer(clean_triangles, op_flags=['readwrite']):
        x[...] = new_order[int(x)]
    self.triangles = clean_triangles
    return self


def filter_triangles(self, keep_list):
    if len(keep_list) == 0:
        raise ValueError('Provided list of nodes is empty.')

    self.triangles = self.triangles[keep_list, :]
    return self


def isolated_vertices(self):
    '''Returns the set of vertices that do not belong in any triangle.
    '''

    referenced = set(self.triangles.ravel())
    if len(referenced) == self.num_vertices:
        return set()
    else:
        return set(np.arange(self.num_vertices)) - referenced


def has_identical_triangles(self):
    new_tr = unique_rows(self.triangles)
    return len(new_tr) != self.num_triangles


def clean_identical_triangles(self, verbose=False):
    new_tr = unique_rows(self.triangles)
    if len(new_tr) != self.num_triangles:
        if verbose:
            print('Identical triangles were detected and are being deleted.')
        self.triangles = new_tr
    return self


def _get_non_duplicate_triangles(self):
    eqc = defaultdict(list)
    for i, row in enumerate(self.triangles):
        eqc[tuple(sorted(row))].append(i)
    return [min(v_id) for v_id in eqc.values()]


def has_duplicate_triangles(self):
    good_tr = _get_non_duplicate_triangles(self)
    return self.num_triangles != len(good_tr)


def clean_duplicate_triangles(self, verbose=False):
    '''Removed the duplicate triangles of a mesh. Two triangles are considered duplicate of each other if they
    reference the same set of vertices.
    '''
    keep_list = _get_non_duplicate_triangles(self)
    if self.num_triangles != len(keep_list):
        if verbose:
            print('Duplicate triangles were detected and are being deleted.')
        self.triangles = self.triangles[keep_list, :]
    return self


def clean_degenerate_triangles(self, verbose=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = self.area_of_triangles()
        good_tr = A > 0
        if np.sum(good_tr) != self.num_triangles:
            if verbose:
                print('Deleting triangles with zero area.')
            self.triangles = self.triangles[good_tr, :]

    assert(all(self.area_of_triangles() > 0))

    A = self.angles_of_triangles()
    bad_triangles = np.where((A == 0).any(axis=1))[0]
    if bad_triangles.size > 0:
        if verbose:
            print('Deleting triangles containing angles that are 0 degrees.')
        keep = list(set(range(self.num_triangles)) - set(bad_triangles))
        self = filter_triangles(self, keep)
    return self


def clean_zero_area_vertices(self, verbose=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        A = self.area_of_vertices()
    bad_vert = np.where(A <= 0)[0]
    if bad_vert.size > 0:
        if verbose:
            print('Deleting vertices with zero area.')
        keep_list = list(set(range(self.num_vertices)) - set(bad_vert))
        self = filter_vertices(self, keep_list)
    assert(all(self.area_of_vertices() > 0))
    return self


def clean_isolated_vertices(self, verbose=False):
    bad_vertices = isolated_vertices(self)
    if bad_vertices:
        if verbose:
            print('Deleting isolated vertices.')
        keep_list = list(set(range(self.num_vertices)) - bad_vertices)
        self = filter_vertices(self, keep_list)
    return self


def clean_identical_vertices(self, verbose=False):
    '''Removes any vertex that has exactly the same (x,y,z)  coordinates with another
    vertex.
    Notes: Let v1 and v2 be two duplicate vertices and v2 being the one that will be removed.
    All the triangles that reference v2, will now reference v1.
    '''
    eqc = defaultdict(list)
    for i, row in enumerate(self.vertices):
        eqc[tuple(row)].append(i)

    check_list = [sorted(c) for c in eqc.values() if len(c) > 1]
    if check_list:
        if verbose:
            print('Duplicate vertices were detected and are being deleted.')

        keep_list = [min(v_id) for v_id in eqc.values()]
        T = self.triangles.ravel()
        for v_id in check_list:
            ix = np.in1d(T, v_id[1:]).reshape(T.shape)
            T[ix] = v_id[0]

        self.triangles = T.reshape(self.triangles.shape)
        filter_vertices(self, keep_list)
        clean_identical_triangles(self)     # TODO - don't clean. Let caller do that.
    return self


def clean_mesh(self, level=3, verbose=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clean_degenerate_triangles(self, verbose)
        if level >= 2:
            clean_zero_area_vertices(self, verbose)
            clean_isolated_vertices(self, verbose)
            clean_identical_triangles(self, verbose)
        if level == 3:
            clean_duplicate_triangles(self, verbose)
    return self
