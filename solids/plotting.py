'''
Created on Dec 30, 2017

@author: optas
'''

import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.graph_objs import Mesh3d, Data, Scatter3d, Line
from plotly.offline import iplot

from .. point_clouds import Point_Cloud


def plot_mesh_via_matplotlib(in_mesh, in_u_sphere=True, axis=None, figure=None, figsize=(5, 5), gspec=111, colormap=cm.viridis, plot_edges=False, vertex_color=None, show=True):
    '''Alternative to plotting a mesh with matplotlib.
       TODO Need colorize vertex/faces. more input options'''

    faces = in_mesh.triangles
    verts = in_mesh.vertices
    if in_u_sphere:
        verts = Point_Cloud(verts).center_in_unit_sphere().points

    if figure is None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = figure

    if axis is None:
        ax = fig.add_subplot(gspec, projection='3d')
    else:
        ax = axis
    
    mesh = Poly3DCollection(verts[faces])
    
    if plot_edges:
        mesh.set_edgecolor('k')

    if vertex_color is not None:
        
        face_color=in_mesh.triangle_weights_from_vertex_weights(vertex_color)
        mappable = cm.ScalarMappable(cmap=colormap)
        colors = mappable.to_rgba(face_color)
        colors[:,3]=1
        mesh.set_facecolor(colors)


    ax.add_collection3d(mesh)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    miv = 0.7*np.min(verts)
    mav = 0.7*np.max(verts)
    ax.set_xlim(miv, mav)
    ax.set_ylim(miv, mav)
    ax.set_zlim(miv, mav)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig


def plot_mesh_via_plottly(in_mesh, colormap=cm.RdBu, plot_edges=None, vertex_color=None, show=True):
    ''' Winnie note: I found most of this code in some online documentation. Unfortunatelly, can't remember where. 
    '''
    x = in_mesh.vertices[:, 0]
    y = in_mesh.vertices[:, 1]
    z = in_mesh.vertices[:, 2]
    simplices = in_mesh.triangles
    tri_vertices = map(lambda index: in_mesh.vertices[index], simplices)    # vertices of the surface triangles
    I, J, K = ([triplet[c] for triplet in simplices] for c in range(3))

    triangles = Mesh3d(x=x, y=y, z=z, i=I, j=J, k=K, name='', intensity=vertex_color)

    if plot_edges is None:  # The triangle edges are not plotted.
        res = Data([triangles])
    else:
        # Define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
        # None separates data corresponding to two consecutive triangles
        lists_coord = [[[T[k % 3][c] for k in range(4)] + [None] for T in tri_vertices] for c in range(3)]
        Xe, Ye, Ze = [reduce(lambda x, y: x + y, lists_coord[k]) for k in range(3)]

        # Define the lines to be plotted
        lines = Scatter3d(x=Xe, y=Ye, z=Ze, mode='lines', line=Line(color='rgb(50, 50, 50)', width=1.5))
        res = Data([triangles, lines])

    if show:
        iplot(res)
    else:
        return res
