'''
Created on July 21, 2016

@author: Panos Achlioptas
@contact: pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from math import ceil

from .. utils import linalg_utils as utils
from .. utils.linalg_utils import l2_norm
from .. solids import mesh_cleaning as cleaning
from .. fundamentals.graph import Graph


class Laplace_Beltrami(object):
    '''A class representing a discretization of the Laplace Beltrami operator, associated with a given
    Mesh object.
    '''
    def __init__(self, in_mesh):
        '''
        Constructor
        Notes: when a duplicate triangle exists for instance it will contribute twice in
        the computation of the area of each of its vertices.
        '''
        if cleaning.has_duplicate_triangles(in_mesh):   # Add test for zero areas. and degenerate triangles cotangent is infinite there).
            raise ValueError('The given mesh contains duplicate triangles. Please clean them before making an LB.')
        self.M = in_mesh
        self.W = Laplace_Beltrami.cotangent_laplacian(self.M)

    def spectra(self, k, area_type='barycentric', normalize=True):
        A = self.M.area_of_vertices(area_type)
        A = sparse.spdiags(A[:, 0], 0, A.size, A.size)
        evals, evecs = eigs(self.W, k, A, sigma=-10e-1, which='LM')

        if np.any(l2_norm(evecs.imag, axis=0) / l2_norm(evecs.real, axis=0) > 1.0 / 100):
            warnings.warn('Produced eigen-vectors are complex and contain significant mass on the imaginary part.')

        evecs = evecs.real   # eigs returns complex values by default.
        evals = evals.real

        nans = np.isnan(evecs)
        if nans.any():
            warnings.warn('NaN values were produced in some evecs. These evecs will be dropped.')
            ok_evecs = np.sum(nans, axis=0) == 0

            if ok_evecs.any():
                evecs = evecs[:, ok_evecs]
                evals = evals[ok_evecs]
            else:
                return []

        gram_matrix = (A.dot(evecs)).T.dot(evecs)
        gram_matrix = gram_matrix - np.eye(evecs.shape[1])
        if np.max(gram_matrix) > 10e-5:
            warnings.warn('Eigenvectors are not orthogonal within  10e-5 relative error.')

        evals, evecs = utils.sort_spectra(evals, evecs)

        return evals, evecs

    def multi_component_spectra(self, k, area_type, percent=None, min_nodes=None, min_eigs=None, max_eigs=None, thres=1):
        _, node_labels = self.M.connected_components()
        cc_at_thres = Graph.largest_connected_components_at_thres(node_labels, thres)
        n_cc = len(cc_at_thres)
        E = list()
        for i in xrange(n_cc):
            keep = cc_at_thres[i]
            temp_mesh = self.M.copy()
            cleaning.filter_vertices(temp_mesh, keep)
            cleaning.clean_mesh(temp_mesh)
            num_nodes = temp_mesh.num_vertices
            if min_nodes is not None and num_nodes < min_nodes:
                E.append([])
                continue
            if percent is not None:
                k = int(ceil(percent * num_nodes))

            if min_eigs is not None:
                k = max(k, min_eigs)

            if max_eigs is not None:
                k = min(k, max_eigs)
            try:
                feasible_k = min(k, num_nodes - 2)
                E.append((Laplace_Beltrami(temp_mesh).spectra(feasible_k, area_type)))
            except:
                print 'Component %d failed.' % (i)
                E.append([])
        return E, cc_at_thres

    @staticmethod
    def cotangent_laplacian(in_mesh):
        '''Computes the cotangent laplacian weight matrix. Also known as the stiffness matrix.
        Output: a PSD matrix.
        '''
        T = in_mesh.triangles
        angles = in_mesh.angles_of_triangles()
        I = np.hstack([T[:, 0], T[:, 1], T[:, 2]])
        J = np.hstack([T[:, 1], T[:, 2], T[:, 0]])
        S = 0.5 / np.tan(np.hstack([angles[:, 2], angles[:, 0], angles[:, 1]]))   # TODO-P Possible division by zero
        In = np.hstack([I, J, I, J])
        Jn = np.hstack([J, I, I, J])
        Sn = np.hstack([-S, -S, S, S])
        W = sparse.csc_matrix((Sn, (In, Jn)), shape=(in_mesh.num_vertices, in_mesh.num_vertices))
        if utils.is_symmetric(W, tolerance=10e-5) == False:
            warnings.warn('Cotangent matrix is not symmetric within epsilon: %f' % (10e-5,))
            W /= 0.5
            W = W + W.T
        return W
