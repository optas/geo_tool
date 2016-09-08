import sys
import os
import re
import os.path as osp
import numpy as np
from itertools import izip
import matplotlib.pylab as plt

from nn_saliency.src.shape_views import Shape_Views
from nn_saliency.src.view_gradients import View_Gradients
from nn_saliency.src.back_tracer import Back_Tracer
# from nn_saliency.geo_tool.solids.mesh import Mesh
from nn_saliency.src import linalg_utils utils
from nn_saliency.src import nn_io

rendered_size = (256, 256)      # FYthumb default output size

if __name__ == '__main__':
    top_view_dir = sys.argv[1]  # Top dir containing folders of views
    top_mesh_dir = sys.argv[2]
    out_top_dir  = sys.argv[3]
    regex = re.compile(sys.argv[4])

    sub_files   = [osp.join(top_view_dir, f) for f in os.listdir(top_view_dir)]
    view_dirs   = [d for d in sub_files if osp.isdir(d)]                    
    model_names = [osp.basename(m) for m in view_dirs]                      
    mesh_files  = [osp.join(top_mesh_dir, m + '.off') for m in model_names]
    
    for views, mesh_file, model_name in izip(view_dirs, mesh_files, model_names):
        if not regex.search(mesh_file):
            continue
                
        print mesh_file, '\n'
#         sv = Shape_Views(views, 'png')
#         raw_grads_file = osp.join(views, 'raw_grads.npz')
#         grads = View_Gradients(sv, raw_grads_file)
#         grads.resize(rendered_size)
#         grads.clean_grad_outside_shape_mask()
#         abs_grads = grads.transform_grads(np.abs)         
        grads_output_dir = osp.join(out_top_dir, 'Abs_Grads', model_name)
#         abs_grads.export_grads_to_txt(grads_output_dir)               
#         in_mesh = Mesh(mesh_file)                                
#         mask_output_dir = osp.join(out_top_dir, 'Masks', model_name)         
#         sv.export_masks_to_txt(mask_output_dir)
        triangle_output_dir = osp.join(out_top_dir, 'Triangles_of_Abs_Grads', model_name)
        nn_io.create_dir(triangle_output_dir)        
        Back_Tracer.compute_triangles_from_pixels(mesh_file, grads_output_dir, triangle_output_dir) 
#         bt = Back_Tracer(triangle_output_dir, in_mesh)
#         grads_on_mesh = abs_grads.push_on_triangles(bt)[0]#         
#         plt.plot(grads_on_mesh)
#         plt.plot(grads_on_mesh / in_mesh.area_of_triangles())#         
#         grads_on_mesh = utils.scale(grads_on_mesh / in_mesh.area_of_triangles() )
#         in_mesh.plot(triangle_function=grads_on_mesh)