'''
Created on Jun 30, 2018

@author: optas
'''

import numpy as np
from . point_cloud import Point_Cloud


def pclouds_with_zero_mean_in_unit_sphere(in_pclouds):
    ''' Zero MEAN + Max_dist = 0.5
    '''
    pclouds = in_pclouds.copy()
    pclouds = pclouds - np.expand_dims(np.mean(pclouds, axis=1), 1)
    dist = np.max(np.sqrt(np.sum(pclouds ** 2, axis=2)), 1)
    dist = np.expand_dims(np.expand_dims(dist, 1), 2)
    pclouds = pclouds / (dist * 2.0)
    return pclouds


def center_pclouds_in_unit_sphere(pclouds):
    for i, pc in enumerate(pclouds):
        pc, _ = Point_Cloud(pc).center_axis()
        pclouds[i] = pc.points

    dist = np.max(np.sqrt(np.sum(pclouds ** 2, axis=2)), 1)
    dist = np.expand_dims(np.expand_dims(dist, 1), 2)
    pclouds = pclouds / (dist * 2.0)

    for i, pc in enumerate(pclouds):
        pc, _ = Point_Cloud(pc).center_axis()
        pclouds[i] = pc.points

    dist = np.max(np.sqrt(np.sum(pclouds ** 2, axis=2)), 1)
    assert(np.all(abs(dist - 0.5) < 0.0001))
    return pclouds
