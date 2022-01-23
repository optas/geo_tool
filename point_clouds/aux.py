'''
Created on Aug 21, 2017

@author: optas
'''
import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import eigs
from numpy.linalg import norm

from .. fundamentals import Graph
from .. utils.linalg_utils import l2_norm

def greedy_match_pc_to_pc(from_pc, to_pc):
    '''map from_pc points to to_pc by minimizing the from-to-to euclidean distance.'''
    nn = NearestNeighbors(n_neighbors=1).fit(to_pc)
    distances, indices = nn.kneighbors(from_pc)
    return indices, distances


def chamfer_pseudo_distance(pc1, pc2):
    _, d1 = greedy_match_pc_to_pc(pc1, pc2)
    _, d2 = greedy_match_pc_to_pc(pc2, pc1)
    return np.sum(d1) + np.sum(d2)


def laplacian_spectrum(pc, n_evecs, k=6):
    ''' k: (int) number of nearest neighbors each point is connected with in the constructed Adjacency
    matrix that will be used to derive the Laplacian.
    '''
    neighbors_ids, distances = pc.k_nearest_neighbors(k)
    A = Graph.knn_to_adjacency(neighbors_ids, distances)
    if Graph.connected_components(A)[0] != 1:
        raise ValueError('Graph has more than one connected component, increase k.')
    A = (A + A.T) / 2.0
    L = Graph.adjacency_to_laplacian(A, 'norm').astype('f4')
    evals, evecs = eigs(L, n_evecs + 1, sigma=-10e-1, which='LM')
    if np.any(l2_norm(evecs.imag, axis=0) / l2_norm(evecs.real, axis=0) > 1.0 / 100):
        warnings.warn('Produced eigen-vectors are complex and contain significant mass on the imaginary part.')

    evecs = evecs.real   # eigs returns complex values by default.
    evals = evals.real

    index = np.argsort(evals)  # Sort evals from smallest to largest
    evals = evals[index]
    evecs = evecs[:, index]
    return evals, evecs


def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    '''Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    '''
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in xrange(resolution):
        for j in xrange(resolution):
            for k in xrange(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    Original from https://github.com/daerduoCarey/partnet_seg_exps/blob/master/exps/utils/pc_util.py
    """
    vol = np.zeros((vsize,vsize,vsize))
    voxel = 2*radius/float(vsize)
    locations = (points + radius)/voxel
    locations = locations.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol


def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    Original from Original from https://github.com/daerduoCarey/partnet_seg_exps/blob/master/exps/utils/pc_util.py
    """
    vsize = vol.shape[0]
    assert(vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a,b,c] == 1:
                    points.append(np.array([a, b, c]))
    if len(points) == 0:
        return np.zeros((0, 3))
    points = np.vstack(points)
    return points