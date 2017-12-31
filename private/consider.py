'''
Created on December 21, 2016.
Keep things that you will consider to include in main library.

@author:    Panos Achlioptas
@contact:   pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any
            way you want for non-commercial purposes.
'''

from mayavi import mlab as mayalab


def plot_vector_field(points, vx, vy, vz):
    mayalab.quiver3d(points[:, 0], points[:, 1], points[:, 2], vx, vy, vz)
    mayalab.show()
