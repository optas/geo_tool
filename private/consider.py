'''
Created on December 21, 2016.
Keep things that you will consider to include in main library.

@author:    Panos Achlioptas
@contact:   pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any
            way you want for non-commercial purposes.
'''

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D


def plot_mesh_2(in_mesh, show=True, in_u_sphere=False):
    '''Alternative to plotting a mesh with matplotlib.
    Need to find way to colorize vertex/faces.'''

    fig = plt.figure()
    ax = Axes3D(fig)
    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    v = in_mesh.vertices
    tri = Poly3DCollection(v[in_mesh.triangles])
    ax.add_collection3d(tri)
    if show:
        plt.show()
    else:
        return fig