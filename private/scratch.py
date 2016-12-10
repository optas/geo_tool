'''
Created on June 14, 2016.
Dirty scripts checking geo_tool functionality 

@author:    Panayotes Achlioptas
@contact:   pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any
            way you want for non-commercial purposes. 
'''
 
import sys
import numpy as np
import os.path as osp
from scipy import spatial
 
 
git_path = '/Users/optas/Documents/Git_Repos/'
sys.path.insert(0, git_path)

from geo_tool import Mesh, Graph, Point_Cloud, Laplace_Beltrami
import geo_tool.solids.mesh_cleaning as cleaning
import geo_tool.signatures.node_signatures as ns


############################################################
## TODO: check ravel() vs. np.repeat (used in geo_tool)
## 
## 
############################################################ 

def main_Mesh():    
    off_file = '/Users/optas/DATA/Shapes/Model_Net_10/OFF_Original/bathtub/train/bathtub_0001.off'
    
    in_mesh = Mesh(off_file=off_file)
    in_mesh.center_in_unit_sphere()
    cleaning.clean_mesh(in_mesh, level=3, verbose=True)
        
    in_lb = Laplace_Beltrami(in_mesh)
    n_cc, node_labels = in_mesh.connected_components()
    parts_id = Graph.largest_connected_components_at_thres(node_labels, 1)
    print 'Number of connected components = %d.' % (n_cc)

    percent_of_eigs = 1
    min_eigs = None
    max_eigs = None
    min_vertices = None
    time_horizon = 10
    area_type = 'barycentric'

    v_color = ns.hks_of_component_spectra(in_mesh, in_lb, area_type, percent_of_eigs, \
                                      time_horizon, min_vertices, min_eigs, max_eigs)[0]

    in_mesh.plot(vertex_function=v_color)


def main_Point_Cloud():
    ply_file = '/Users/optas/Documents/Git_Repos/point_cloud_saliency/test_data/airplane.ply'
    cloud = Point_Cloud(ply_file=ply_file)
    print cloud



if __name__ == '__main__':
    main_Mesh()
    main_Point_Cloud()