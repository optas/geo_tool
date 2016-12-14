'''
Created on July 18, 2016

@author: Panayotes Achlioptas
@contact: pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

import itertools
import warnings
import copy
import numpy as np
from scipy import sparse as sp
from numpy.matlib import repmat
from mayavi import mlab as mayalab

import mesh_cleaning as cleaning
from .. utils import linalg_utils as utils
from .. utils.linalg_utils import accumarray
from .. in_out import soup as io
from .. fundamentals import Graph, Cuboid
from .. point_clouds import Point_Cloud

l2_norm = utils.l2_norm


class Mesh(object): 
    '''
    A class representing a triangular Mesh of a 3D surface. Provides a variety of relevant functions, including
    loading and plotting utilities.
    '''
    def __init__(self, vertices=None, triangles=None, off_file=None):
        '''
        Constructor
        '''
        if off_file != None:
            self.vertices, self.triangles = io.load_off(off_file)[:2]
        else:
            self.vertices = vertices 
            self.triangles = triangles
            
    @property
    def vertices(self):
        return self._vertices

    @property
    def triangles(self):
        return self._triangles

    @vertices.setter
    def vertices(self, value):
        self._vertices = value
        self.num_vertices = len(self._vertices)

    @triangles.setter
    def triangles(self, value):
        self._triangles = value
        self.num_triangles = len(self._triangles)
        if not all([len(set(tr)) == 3 for tr in self._triangles]):
            warnings.warn('Not real triangles (but lines or points) exist in the triangle list.')
        if np.max(self._triangles) > self.num_vertices - 1 or np.min(self._triangles) < 0:
            raise ValueError('Triangles referencing non-vertices.')

    def __str__(self):
        return 'Mesh with %d vertices and %d triangles.' % (self.num_vertices, self.num_triangles)

    def copy(self):
        return copy.deepcopy(self)

    def plot(self, triangle_function=np.array([]), vertex_function=np.array([]), show=True, *args, **kwargs):
        if vertex_function.any() and triangle_function.any():
            raise ValueError('Either the vertices or the faces can mandate color. Not both.')

        if triangle_function.any():
            if len(triangle_function) != self.num_triangles:
                raise ValueError('Function of triangles has inappropriate number of elements.')
            mesh_plot = mayalab.triangular_mesh(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], self.triangles, *args, **kwargs)
            self.__decorate_mesh_with_triangle_color(mesh_plot, triangle_function)

        elif vertex_function.any():
            if len(vertex_function) != self.num_vertices:
                raise ValueError('Function of vertices has inappropriate number of elements.')
            mesh_plot = mayalab.triangular_mesh(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], self.triangles, scalars=vertex_function, *args, **kwargs)

        else:
            mesh_plot = mayalab.triangular_mesh(self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2], self.triangles, *args, **kwargs)

        if show:
            mayalab.show()
        else:
            return mesh_plot

    def plot_normals(self, scale_factor=1, representation='mesh'):
        self.plot(show=False, representation=representation)
        bary = self.barycenter_of_triangles()
        normals = Mesh.normals_of_triangles(self.vertices, self.triangles)
        mayalab.quiver3d(bary[:, 0], bary[:, 1], bary[:, 2], normals[:, 0], normals[:, 1], normals[:, 2], scale_factor=scale_factor)
        mayalab.show()

    def undirected_edges(self):
        # TODO - avoid double counting (see unique_rows) + make a version for directed
        perm_gen = lambda x: list(itertools.permutations(x, 2))
        edges = np.zeros(shape=(self.num_triangles, 6, 2), dtype=np.int32)  # Each triangle produces 6 undirected edges.
        for i, t in enumerate(self.triangles):
            edges[i, :] = perm_gen(t)
        edges.resize(self.num_triangles * 6, 2)
        return utils.unique_rows(edges)

    def adjacency_matrix(self):
        E = self.undirected_edges()
        vals = np.squeeze(np.ones((len(E), 1)))
        return sp.csr_matrix((vals, (E[:, 0], E[:, 1])), shape=(self.num_vertices, self.num_vertices))

    def connected_components(self):
        return Graph.connected_components(self.adjacency_matrix())    
    
    def barycenter_of_triangles(self):
        tr_in_xyz = self.vertices[self.triangles]
        return np.sum(tr_in_xyz, axis=1) / 3.0

    def edge_length_of_triangles(self):
        '''Computes the length of each edge, of each triangle in the underlying triangular mesh.

        Returns:
            L - (num_of_triangles x 3) L[i] is a triple containing the lengths of the 3 edges corresponding to the i-th triangle.
            The enumeration of the triangles is the same at in -T- and the order in which the edges are
            computed is (V2, V3), (V1, V3) (V1, V2). I.e. L[i][2] is the edge length between the 1st
            vertex and the third vertex of the i-th triangle.'''
        V = self.vertices
        T = self.triangles
        L1 = l2_norm(V[T[:, 1], :] - V[T[:, 2], :], axis=1)
        L2 = l2_norm(V[T[:, 0], :] - V[T[:, 2], :], axis=1)
        L3 = l2_norm(V[T[:, 0], :] - V[T[:, 1], :], axis=1)
        return np.transpose(np.vstack([L1, L2, L3]))

    def inverse_triangle_dictionary(self):
        '''Returns a dictionary mapping triangles, i.e., triplets (n1, n2, n3) into their
        position in the array of triangles kept
        '''
        keys = map(tuple, self.triangles)
        return dict(zip(keys, range(len(keys))))

    def angles_of_triangles(self):
        # TODO: Consider compute via way mentioned in Meyer's
        L = self.edge_length_of_triangles()
        L1 = L[:, 0]
        L2 = L[:, 1]
        L3 = L[:, 2]

        L1_sq = np.square(L1)
        L2_sq = np.square(L2)
        L3_sq = np.square(L3)

        A1 = (L2_sq + L3_sq - L1_sq) / (2. * L2 * L3)    # Cosine of angles for first set of edges.
        A2 = (L1_sq + L3_sq - L2_sq) / (2 * L1 * L3)
        A3 = (L1_sq + L2_sq - L3_sq) / (2 * L1 * L2)
        A = np.transpose(np.vstack([A1, A2, A3]))

        if np.any(A <= -1) or np.any(A >= 1) or (np.isfinite(A) == False).any():
            warnings.warn('The mesh has degenerate triangles with angles outside the (0,pi) interval. This angles will be set to 0.')
            A[np.logical_or(A >= 1, A <= -1, np.isfinite(A) == False)] = 1

        A = np.arccos(A)
        assert(np.all(np.logical_and(A < np.pi, A >= 0)))
        return A

    def area_of_triangles(self):
        '''Computes the area of each triangle, in a triangular mesh.
        '''
        A = Mesh.normals_of_triangles(self.vertices, self.triangles)
        A = l2_norm(A, axis=1) / 2.0
        if np.any(A <= 0):
            warnings.warn('The mesh has triangles with non positive area.')
        return A

    def area_of_vertices(self, area_type='barycentric'):
        '''
            area_type == 'barycentric' associates with every vertex the area of its adjacent barycentric cells.
                         'barycentric_avg' same as 'barycentric' but post multiplied with the adjacency matrix. I.e.,
                         each node is assigned the average of the barycentric areas of it's neighboring nodes.
        '''
        def barycentric_area():
            I = np.hstack([T[:, 0], T[:, 1], T[:, 2]])
            J = np.hstack([T[:, 1], T[:, 2], T[:, 0]])
            Mij = (1.0 / 12) * np.hstack([Ar, Ar, Ar])
            Mji = np.copy(Mij)
            Mii = (1.0 / 6) * np.hstack([Ar, Ar, Ar])
            In = np.hstack([I, J, I])
            Jn = np.hstack([J, I, I])
            Mn = np.hstack([Mij, Mji, Mii])
            M = sp.csr_matrix((Mn, (In, Jn)), shape=(self.num_vertices, self.num_vertices))
            M = np.array(M.sum(axis=1))
            return M

        Ar = self.area_of_triangles()
        T = self.triangles

        if area_type == 'barycentric':
            M = barycentric_area()
        elif area_type == 'barycentric_avg':
            M = self.adjacency_matrix() * barycentric_area()
        else:
            raise(NotImplementedError)

        if np.any(M <= 0):
                warnings.warn('The area_type \'%s\' produced vertices with non-positive area.' % (area_type))
        return M

    def sum_vertex_function_on_triangles(self, v_func):
        if len(v_func) != self.num_vertices:
            raise ValueError('Provided vertex function has inappropriate dimensions. ')
        tr_func = np.zeros((self.num_triangles, 1))
        for i, tr in enumerate(self.triangles):
            v1, v2, v3 = tr
            tr_func[i] = v_func[v1] + v_func[v2] + v_func[v3]
        return tr_func

    def normals_of_vertices(self, weight='areas', normalize=False):
        '''Computes the outward normal at each vertex by adding the weighted normals of each triangle a
        vertex is adjacent to. The weights that are used to combine the normals are the areas of the triangles
        a normal comes from.

        Args:
            normalize (boolean): if True, then the normals have unit length.

        Returns:
            N   -   (num_of_vertices x 3) an array containing the normalized outward normals of all the vertices.
        '''
        V = self.vertices
        T = self.triangles
        normals = Mesh.normals_of_triangles(V, T)

        if weight == 'areas':
            weights = self.area_of_triangles()
            normals = (normals.T * weights).T

        nx = accumarray(T.ravel('C'), repmat(normals[:, 0], 1, 3).ravel('C'))
        ny = accumarray(T.ravel('C'), repmat(normals[:, 1], 1, 3).ravel('C'))
        nz = accumarray(T.ravel('C'), repmat(normals[:, 2], 1, 3).ravel('C'))
        normals = (np.vstack([nx, ny, nz])).T

        if normalize:
            row_norms = l2_norm(normals, axis=1)
            row_norms[row_norms == 0] = 1
            normals = (normals.T / row_norms).T

        return normals

    def bounding_box(self):
        return Cuboid.bounding_box_of_3d_points(self.vertices)
    
    def center_in_unit_sphere(self):
        self.vertices = Point_Cloud.center_points_in_unit_sphere(self.vertices)
        return self
    
    def sample_faces(self, n_samples):
        """Generates a point cloud representing the surface of the mesh by sampling points
        proportionally to the area of each face.

        Args:
            n_samples (int) : number of points to be sampled in total

        Returns:
            numpy array (n_samples, 3) containing the [x,y,z] coordinates of the samples.

        Reference :
          http://chrischoy.github.in_out/research/barycentric-coordinate-for-mesh-sampling/
          [1] Barycentric coordinate system

          \begin{align}
            P = (1 - \sqrt{r_1})A + \sqrt{r_1} (1 - r_2) B + \sqrt{r_1} r_2 C
          \end{align}
        """
        face_areas = self.area_of_triangles()
        face_areas = face_areas / np.sum(face_areas)

        n_samples_per_face = np.ceil(n_samples * face_areas)
        n_samples_per_face = n_samples_per_face.astype(np.int)
        n_samples = int(np.sum(n_samples_per_face))

        # Create a vector that contains the face indices
        sample_face_idx = np.zeros((n_samples, ), dtype=int)

        acc = 0
        for face_idx, _n_sample in enumerate(n_samples_per_face):
            sample_face_idx[acc: acc + _n_sample] = face_idx
            acc += _n_sample

        r = np.random.rand(n_samples, 2)
        A = self.vertices[self.triangles[sample_face_idx, 0], :]
        B = self.vertices[self.triangles[sample_face_idx, 1], :]
        C = self.vertices[self.triangles[sample_face_idx, 2], :]
        P = (1 - np.sqrt(r[:, 0:1])) * A + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + \
            np.sqrt(r[:, 0:1]) * r[:, 1:] * C
        return P, sample_face_idx

    @staticmethod
    def __decorate_mesh_with_triangle_color(mesh_plot, triangle_function):   # TODO-P do we really need this to be static?
        mesh_plot.mlab_source.dataset.cell_data.scalars = triangle_function
        mesh_plot.mlab_source.dataset.cell_data.scalars.name = 'Cell data'
        mesh_plot.mlab_source.update()
        mesh2 = mayalab.pipeline.set_active_attribute(mesh_plot, cell_scalars='Cell data')
        mayalab.pipeline.surface(mesh2)

    @staticmethod
    def normals_of_triangles(V, T, normalize=False):
        '''Computes the normal vector of each triangle of a given mesh.
        Args:
            V           -   (num_of_vertices x 3) 3D coordinates of the mesh vertices.
            T           -   (num_of_triangles x 3) T[i] are the 3 indices corresponding to the 3 vertices of
                            the i-th triangle. The indexing is based on -V-.
            normalize   -   (Boolean, optional) if True, the normals will be normalized to have unit lenght.

        Returns:
            N           -   (num_of_triangles x 3) an array containing the outward normals of all the triangles.
        '''
        N = np.cross(V[T[:, 0], :] - V[T[:, 1], :], V[T[:, 0], :] - V[T[:, 2], :])
        if normalize:
            row_norms = l2_norm(N, axis=1)
            N = (N.T / row_norms).T
        return N