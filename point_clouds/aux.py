'''
Created on Aug 21, 2017

@author: optas
'''
import numpy as np
from sklearn.neighbors import NearestNeighbors


def greedy_match_pc_to_pc(from_pc, to_pc):
    '''map from_pc points to to_pc by minimizing the from-to-to euclidean distance.'''
    nn = NearestNeighbors(n_neighbors=1).fit(to_pc)
    distances, indices = nn.kneighbors(from_pc)
    return indices, distances


def chamfer_pseudo_distance(pc1, pc2):
    _, d1 = greedy_match_pc_to_pc(pc1, pc2)
    _, d2 = greedy_match_pc_to_pc(pc2, pc1)
    return np.sum(d1) + np.sum(d2)
