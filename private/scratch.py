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
from geo_tool import Mesh, Point_Cloud
 
 
ply_file = '/Users/optas/Documents/Git_Repos/point_cloud_saliency/test_data/airplane.ply'
cloud = Point_Cloud(ply_file=ply_file)
print cloud


# 
# import geo_tool.signatures.node_signatures as ns
# import geo_tool.utils.linalg_utils as la_utils
# import geo_tool.in_out.soup as io
# import geo_tool.solids.mesh_cleaning as cleaning
# # from geo_tool.solids.mesh_cleaning
# 
# l2_norm = np.linalg.norm



# # TODO: check ravel() vs. np.repeat.
# from solids.mesh import Mesh
# import solids.mesh_cleaning as mc 
# # import os.path as osp
# 
# 
# DATA_DIR = '/Users/optas/Documents/DATA/Model_Net_10'
# class_type = 'bath_tub'
# model_id = 0001
# 
# # off_file = osp.join(DATA_DIR, class_type, 'train', model_id, '.off')
# 
# off_file = '/Users/optas/Documents/DATA/Model_Net_10/OFF_Original/bathtub/train/bathtub_0001.off'
# 
# in_mesh = Mesh(off_file)
# mc.clean_mesh(in_mesh, level=3, verbose=True)
# in_mesh.plot()

# if __name__ == '__main__' and __package__ is None:
