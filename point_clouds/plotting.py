'''
Created on July 11, 2017

@author: optas
'''

import matplotlib.pyplot as plt
from . point_cloud import Point_Cloud


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
