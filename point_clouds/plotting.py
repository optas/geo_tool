'''
Created on July 11, 2017

@author: optas
'''
import numpy
import matplotlib.pyplot as plt
try:
    from mayavi import mlab as mayalab
except:
    print 'mayavi not installed.'
    
from mpl_toolkits.mplot3d import Axes3D

from . point_cloud import Point_Cloud

l2_norm = numpy.linalg.norm


def plot_pclouds_on_grid(pclouds, grid_size, fig_size=(50, 50), plot_kwargs={}):
    '''Input
            pclouds: Iterable holding point-cloud data. pclouds[i] must be a 2D array with any number of rows and 3 columns.
    '''
    fig = plt.figure(figsize=fig_size)
    c = 1
    for cloud in pclouds:
        plt.subplot(grid_size[0], grid_size[1], c, projection='3d')
        plt.axis('off')
        ax = fig.axes[c - 1]
        Point_Cloud(points=cloud).plot(axis=ax, show=False, **plot_kwargs)
        c += 1
    return fig


def plot_vector_field_mayavi(points, vx, vy, vz):
    mayalab.quiver3d(points[:, 0], points[:, 1], points[:, 2], vx, vy, vz)
    mayalab.show()


def plot_vector_field_matplotlib(pcloud, vx, vy, vz, normalize=True, length=0.01):
    fig = plt.figure()
    ax = Axes3D(fig)
    pts = pcloud.points
    if normalize:
        row_norms = l2_norm(pts, axis=1)
        pts = pts.copy()
        pts = (pts.T / row_norms).T
    return ax.quiver3D(pts[:, 0], pts[:, 1], pts[:, 2], vx, vy, vz, length=length)
