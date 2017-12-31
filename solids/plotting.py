'''
Created on Dec 30, 2017

@author: optas
'''


import matplotlib.pylab as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .. point_clouds import Point_Cloud


def plot_mesh_via_matplotlib(in_mesh, in_u_sphere=True, show=True):
    '''Alternative to plotting a mesh with matplotlib.
       TODO Need colorize vertex/faces. more input options'''

    faces = in_mesh.triangles
    verts = in_mesh.vertices
    if in_u_sphere:
        verts = Point_Cloud(verts).center_in_unit_sphere().points

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    miv = np.min(verts)
    mav = np.max(verts)
    ax.set_xlim(miv, mav)
    ax.set_ylim(miv, mav)
    ax.set_zlim(miv, mav)
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig
