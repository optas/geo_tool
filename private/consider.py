'''
Created on December 21, 2016.
Keep things that you will consider to include in main library. 

@author:    Panas Achlioptas
@contact:   pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any
            way you want for non-commercial purposes. 
'''
# Alternative to plotting a mesh with matplotlib.
# Need to find way to colorize vertex/faces.
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# fig = plt.figure()
# ax = Axes3D(fig)
# v = in_mesh.vertices
# for t in in_mesh.triangles:
#     tri = Poly3DCollection([v[t]])
#     tri.set_color(colors.rgb2hex([0,1,0]))
#     ax.add_collection3d(tri)
# # # # plt.show()


import matplotlib.cm as cm
import matplotlib as mpl

def getRGBA(vec):
    '''
    This function gets a numpy vector as input. It returns a numpy array
    with the same number of rows, but with 4 columns. The colomns correspond
    to RGBA values
    '''
    vmin = 1
    vmax = 0

    cmap = cm.Greens
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    m = cm.ScalarMappable(cmap=cmap)

    return m.to_rgba(vec, bytes=True)