'''
Created on December 8, 2016

@author: Panos Achlioptas
@contact: pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''


import copy
import pickle as pickler
import warnings
import numpy as np
from scipy.linalg import eigh
from numpy.matlib import repmat

try:
    from sklearn.neighbors import NearestNeighbors
except:
    warnings.warn('Sklearn library is not installed.')

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except:
    warnings.warn('Pyplot library is not working correctly. Some graphics utilities will be disabled.')

from .. external_code.python_plyfile.plyfile import PlyElement, PlyData
from .. in_out import soup as io
from .. utils import linalg_utils as utils
from .. fundamentals import Cuboid

l2_norm = utils.l2_norm


class Point_Cloud(object):
    '''
    A class representing a 3D Point Cloud.
    Dependencies:
        1. plyfile 0.4: PLY file reader/writer. DOI: https://pypi.python.org/pypi/plyfile
    '''
    def __init__(self, points=None, ply_file=None):
        '''
        Constructor
        '''
        if ply_file is not None:
            self.points = io.load_ply(ply_file)
        else:
            self.points = points

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._points = value
        self.num_points = len(self._points)

    def __str__(self):
        return 'Point Cloud with %d points.' % (self.num_points)

    def save(self, file_out):
        with open(file_out, "w") as f_out:
            pickler.dump(self, f_out)

    def copy(self):
        return copy.deepcopy(self)

    def permute_points(self, permutation):
        if len(permutation) != 3 or not np.all(np.equal(sorted(permutation), np.array([0, 1, 2]))):
            raise ValueError()
        self.points = self.points[:, permutation]
        return self

    def sample(self, n_samples, replacement=False):
        if n_samples > self.num_points:
            replacement = True
        rindex = np.random.choice(self.num_points, n_samples, replace=replacement)
        return Point_Cloud(points=self.points[rindex, :]), rindex

    def apply_mask(self, bool_mask):
        return Point_Cloud(self.points[bool_mask, :])

    def bounding_box(self):
        return Cuboid.bounding_box_of_3d_points(self.points)

    def center_in_unit_sphere(self):
        self.points = Point_Cloud.center_points_in_unit_sphere(self.points)
        return self

    def plot(self, show=True, in_u_sphere=False, axis=None, *args, **kwargs):
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]
        return Point_Cloud.plot_3d_point_cloud(x, y, z, show=show, in_u_sphere=in_u_sphere, axis=axis, *args, **kwargs)

    def barycenter(self):
        n_points = self.points.shape[0]
        return np.sum(self.points, axis=0) / n_points

    def lex_sort(self, axis=-1):
        '''Sorts the list storing the points of the Point_Cloud in a lexicographical order.
        See numpy.lexsort
        '''
        lex_indices = np.lexsort(self.points.T, axis=axis)
        self.points = self.points[lex_indices, :]
        return self, lex_indices

    def k_nearest_neighbors(self, k):
        #     TODO: Add kwargs of sklearn function
        nn = NearestNeighbors(n_neighbors=k + 1).fit(self.points)
        distances, indices = nn.kneighbors(self.points)
        return indices[:, 1:], distances[:, 1:]

    def normals_lsq(self, k, unit_norm=False):
        '''Least squares normal estimation from point clouds using PCA.
        Args:
                k  (int) indicating how many neighbors the normal estimation is based upon.

        DOI: H. Hoppe, T. DeRose, T. Duchamp, J. McDonald, and W. Stuetzle.
        Surface reconstruction from unorganized points. In Proceedings of ACM Siggraph, pages 71:78, 1992.
        '''
        neighbors, _ = self.k_nearest_neighbors(k)
        points = self.points
        n_points = self.num_points
        N = np.zeros([n_points, 3])
        for i in xrange(n_points):
            x = points[neighbors[i], :]
            p_bar = (1.0 / k) * np.sum(x, axis=0)
            P = (x - repmat(p_bar, k, 1))
            P = (P.T).dot(P)
            [L, E] = eigh(P)
            idx = np.argmin(L)
            N[i, :] = E[:, idx]
        if unit_norm:
            row_norms = np.linalg.norm(N, axis=1)
            N = (N.T / row_norms).T
        return N

    def save_as_ply(self, file_out, normals=None):
        if normals is None:
            vp = np.array([(p[0], p[1], p[2]) for p in self.points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        else:
            values = np.hstack((self.points, normals))
            vp = np.array([(v[0], v[1], v[2], v[3], v[4], v[5]) for v in values], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])

        el = PlyElement.describe(vp, 'vertex')
        PlyData([el]).write(file_out + '.ply')

    @staticmethod
    def center_points_in_unit_sphere(points, ret_trans=False):
        n_points = points.shape[0]
        points_test = points.copy()
        barycenter = np.sum(points, axis=0) / n_points
        points -= barycenter   # Center it in the origin.
        max_dist = np.max(l2_norm(points, axis=1))  # Make max distance equal to one.
        points /= max_dist * 2

        if ret_trans:
            trans_matrix = np.zeros([4, 4])
            trans_matrix[0, 0] = 1.0 / max_dist * 2
            trans_matrix[1, 1] = 1.0 / max_dist * 2
            trans_matrix[2, 2] = 1.0 / max_dist * 2
            trans_matrix[3, 3] = 1.0
            trans_matrix[3, 0] = -1.0 / max_dist * 2 * barycenter[0]
            trans_matrix[3, 1] = -1.0 / max_dist * 2 * barycenter[1]
            trans_matrix[3, 2] = -1.0 / max_dist * 2 * barycenter[2]
            points_test = np.dot(points_test.T, trans_matrix).T
            print(np.allclose(points_test, points))
            return points, trans_matrix
        else:
            return points

    @staticmethod
    def plot_3d_point_cloud(x, y, z, show=True, in_u_sphere=False, axis=None, *args, **kwargs):
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = axis
            fig = axis

        ax.scatter(x, y, z, *args, **kwargs)

        if in_u_sphere:
            ax.set_xlim3d(-0.5, 0.5)
            ax.set_ylim3d(-0.5, 0.5)
            ax.set_zlim3d(-0.5, 0.5)

        if show:
            plt.show()
        return fig

    @staticmethod
    def load(in_file):
        with open(in_file, 'r') as f_in:
            res = pickler.load(f_in)
        return res
