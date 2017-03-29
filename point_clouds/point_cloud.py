'''
Created on December 8, 2016

@author: Panos Achlioptas
@contact: pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''


import copy
import warnings
import numpy as np
from scipy.linalg import eigh
from numpy.matlib import repmat
from six.moves import cPickle

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
        with open(file_out, "wb") as f_out:
            cPickle.dump(self, f_out, protocol=2)

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
        return np.mean(self.points, axis=0)

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

    def rotate_z_axis_by_degrees(self, theta):
        theta = np.deg2rad(theta)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.array([[cos_t, -sin_t, 0],
                      [sin_t, cos_t, 0],
                      [0, 0, 1]])
        self.points = self.points.dot(R)
        return self

    def center_axis(self, axis=None):
        '''Makes the point-cloud to be equally spread around zero on the particular axis, i.e., to be centered. If axis is None, it centers it in all (x,y,z) axis.
        '''
        if axis is None:
            _, g0 = self.center_axis(axis=0)
            _, g1 = self.center_axis(axis=1)
            _, g2 = self.center_axis(axis=2)
            return self, [g0, g1, g2]
        else:

            r_max = np.max(self.points[:, axis])
            r_min = np.min(self.points[:, axis])
            gap = (r_max + r_min) / 2.0
            self.points[:, axis] -= gap
            return self, gap

    def save_as_ply(self, file_out, normals=None, binary=True):
        if normals is None:
            vp = np.array([(p[0], p[1], p[2]) for p in self.points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        else:
            values = np.hstack((self.points, normals))
            vp = np.array([(v[0], v[1], v[2], v[3], v[4], v[5]) for v in values], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])

        el = PlyElement.describe(vp, 'vertex')
        text = not binary
        PlyData([el], text=text).write(file_out + '.ply')

    def is_in_unit_sphere(self, epsilon=10e-5):
        return np.max(l2_norm(self.points, axis=1)) <= (0.5 + epsilon)

    def is_centered_in_origin(self, epsilon=10e-5):
        '''True, iff the extreme values (min/max) of each axis (x,y,z) are symmetrically placed
        around the origin.
        '''
        return np.all(np.max(self.points, 0) + np.min(self.points, 0) < epsilon)

    @staticmethod
    def center_points_in_unit_sphere(points, epsilon=10e-5):
        pc = Point_Cloud(points)

        if not pc.is_in_unit_sphere(epsilon=epsilon):
            max_dist = np.max(l2_norm(points, axis=1))  # Make max distance equal to one.
            pc.points /= (max_dist * 2.0)

        if not pc.is_centered_in_origin(epsilon=epsilon):
            pc.center_axis()

        return pc.points

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
        with open(in_file, 'rb') as f_in:
            res = cPickle.load(f_in)
        return res
