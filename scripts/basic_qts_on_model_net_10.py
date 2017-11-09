'''
Created on Jun 14, 2016

@author:    Panos Achlioptas
@contact:   pachlioptas@gmail.com
@copyright: You are free to use, change, or redistribute this code in any
            way you want for non-commercial purposes. 
'''

import os.path as osp

from nn_saliency.src.mesh import Mesh
import nn_saliency.src.nn_io as nn_io

class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 
               'night_stand', 'sofa', 'table', 'toilet' ]


def connected_components_per_category(top_dir):
    res = dict()
    for category in class_names:
        category_dict = dict()
        look_at   = osp.join(top_dir, category)      
        off_files = nn_io.files_in_subdirs(look_at, '.off$')
        for model in off_files:            
            in_mesh = Mesh(model)
            print in_mesh
            model_name = osp.basename(model)
            n_cc, _ = in_mesh.connected_components()        
            category_dict[model_name] = n_cc             
        res[category] = category_dict            
    return res
        
        