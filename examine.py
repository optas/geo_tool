'''
Created on Aug 27, 2016

@author: optas
'''

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