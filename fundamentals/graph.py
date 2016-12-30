'''
Created on August 3, 2016

@author: Panos Achlioptas
@contact: pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

import numpy as np
from scipy import sparse as sparse
from numpy.matlib import repmat

from .. utils import linalg_utils as lu

class Graph(object):
    '''A class offering some basic graph-related functions. It uses mostly scipy modules.  
    '''

    def __init__(self, params):
        '''
        Constructor
        '''
    
    @staticmethod
    def connected_components(A):    
        return sparse.csgraph.connected_components(A, directed=False)
        
    @staticmethod    
    def largest_connected_components_at_thres(node_labels, thres):
        '''
        Marks the nodes that exist in the largest connected components so that [thres] percent of total nodes are covered.
        '''
        if thres <= 0 or thres > 1:
            raise ValueError('threshold variable must be in (0,1].')        
                
        unique_labels, counts = np.unique(node_labels, return_counts=True)
        decreasing_index = [np.argsort(counts)[::-1]]
        counts = counts[decreasing_index]
        unique_labels = unique_labels[decreasing_index]
        cumulative  = np.cumsum(counts, dtype=np.float32) 
        cumulative /= cumulative[-1] 
        n_cc = max(1, np.sum(cumulative <= thres))        
        cc_marked_list = [np.where(node_labels == unique_labels[i])[0] for i in range(n_cc)]
        assert(any([len(x)<=0 for x in cc_marked_list]) == False)
        return cc_marked_list 
    
    @staticmethod
    def knn_to_adjacency(neighbors, weights, direction='out'):            
        '''Converts neighborhood-like data into the adjacency matrix of an underlying graph.
     
         Args:                
                neighbors  - (N x K) neighbors(i,j) is j-th neighbor of the i-th node.
    
                weights    - (N x K) weights(i,j) is the weight of the (directed) edge between i to j.
    
                direction  - (optional, String) 'in' or 'out'. If 'in' then weights(i,j) correspond to an edge
                             that points towards i. Otherwise, towards j. Default = 'out'.
    
         Returns:   
                   A       - (N x N) sparse adjacency matrix, (i,j) entry corresponds to an edge from i to j.
        '''
            
        if np.any(weights < 0):
            raise ValueError('Non negative weights for an adjacency matrix are not supported.')
                
        n, k = neighbors.shape
        temp = repmat(np.arange(n), k, 1).T
        i = temp.ravel()
        j = neighbors.ravel()
        v = weights.ravel()
        
        A = sparse.csr_matrix((v, (i, j)), shape=(n, n))
        if direction == 'in':
            A = A.T                
        return A
    
    
    @staticmethod
    def adjacency_to_laplacian(A, laplacian_type='comb'):    
        '''Computes the laplacian matrix for a graph described by its adjacency matrix.
          
         Args:    A               - (n x n) Square symmetric adjacency matrix of a graph with n nodes.
                 laplacian_type   - (String, optional) Describes the desired type of laplacian.
        
                                   Valid values:                           
                                       'comb' - Combinatorial (unormalized) laplacian (Default value).
                                       'norm' - Symmetric Normalized Laplacian.
                                       'sign' - Signless Laplacian.
                                               
         Output:   L               - (n x n) sparse matrix of the corresponding laplacian.
                                    
         Notes:  
               DOI: "A Tutorial on Spectral Clustering, U von Luxburg".
        
         (c) Panos Achlioptas 2015  -  http://www.stanford.edu/~optas/FmapLib
        '''
        
        if not lu.is_symmetric(A):
            raise ValueError('Laplacian implemented only for square and symmetric adjacency matrices.')        
        
        n = A.shape[1]
        total_weight = A.sum(axis=1).squeeze()
        D = sparse.spdiags(total_weight, 0, n, n)
                            
        if laplacian_type == 'comb':         
            L = -A + D              
        elif laplacian_type == 'norm':            
            total_weight = (1 / np.sqrt(total_weight)).squeeze()                
            Dn = sparse.spdiags(total_weight, 0, n, n)                                
            L = Dn.dot(-A + D).dot(Dn)
        elif laplacian_type == 'sign':         
            L = A + D        
        else:
            raise ValueError('Please provide a valid argument for the type of laplacian.')
        return L
    
if __name__ == '__main__':
    from geo_tool.solids import mesh_cleaning as cleaning
    from geo_tool.solids.mesh import Mesh
        
    off_file = '/Users/t_achlp/Documents/DATA/ModelNet10/OFF_Original/bathtub/train/bathtub_0001.off'
    in_mesh = Mesh(off_file)
    in_mesh.center_in_unit_sphere()
    cleaning.clean_mesh(in_mesh, level=3, verbose=False)
    
    cloud_points, face_ids = in_mesh.sample_faces(2000)
    from scipy import spatial
    tree = spatial.KDTree(cloud_points, leafsize = 100)
    weights, neighbors = tree.query(cloud_points, 10)
    weights = weights[:,1:]
    neighbors = neighbors[:,1:]    
    weights = np.exp(- weights**2 / (2*np.median(weights)) )
    A = Graph.knn_to_adjacency(neighbors, weights)    
    