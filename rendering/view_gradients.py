'''
Created on Jul 25, 2016

@author: Panos Achlioptas
@contact: pachlioptas@gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes. 
'''
import cv2
import copy
import os.path as osp
import numpy as np
import matplotlib.pylab as plt

from . collections import defaultdict
from . shape_views import Shape_Views
from . back_tracer import Back_Tracer
from .. in_out import soup as nn_io
from .. mesh import Mesh


class View_Gradients(object):
    '''
    classdocs
    '''
    def __init__(self, shape_views, grad_file):
        '''
        Constructor
        '''
        self.shape_views = shape_views
        self.grads       = np.squeeze(np.load(grad_file)['arr_0'])
                
        if self.shape_views.num_views() != self.size(): # gradients is a tensor whose first dim==nun_views
            raise ValueError
    
    def size(self):
        return self.grads.shape[0]
         
    def __iter__(self):
        return self.grads.__iter__()
    
    def __next__(self):
        return self.grads.__next__()
        
    def camera_position(self, grad_id):
        return self.shape_views.cam_pos[grad_id]
    
    def resize(self, new_size):
        new_grads = np.empty(shape=((self.size(),)  + new_size), dtype=self.grads.dtype)        
        for i, old_grad in enumerate(self):
            new_grads[i] = cv2.resize(old_grad, new_size)                                              
        self.grads = new_grads
        
    def clean_grad_outside_shape_mask(self, inline=True):
        if inline:
            new_self = self            
        else:
            new_self = self.copy() 
                
        views_mask = new_self.shape_views.masks      
        dims       = views_mask.shape        
        grad_mask_cleaned = new_self.grads.flatten()   # TODO this is pointer, right?
#         mass_outside = sum(abs(grad_mask_cleaned[~views_mask.flatten()]))
#         mass_inside  = sum(abs(grad_mask_cleaned[views_mask.flatten()]))
#         print mass_outside, mass_inside
        grad_mask_cleaned[views_mask.flatten()==0] = 0
        grad_mask_cleaned = np.reshape(grad_mask_cleaned, dims) 
        new_self.grads = grad_mask_cleaned         
        return new_self
    
    def transform_grads(self, transformer):
        new_grads = self.copy()
        new_grads.grads = transformer(new_grads.grads) 
        return new_grads
                    
    def plot_grad(self, vertex_id, twist_id):
        index = self.shape_views.inv_dict[(vertex_id, twist_id)]
        plt.imshow(self.grads[index,:,:])
        plt.show()
        
    def copy(self):
        new_self = copy.copy(self)                      # Shallow copy all attributes (grads and shape_views) 
        new_self.grads = copy.deepcopy(self.grads)      # Deep copy the grads.     
        return new_self
        
    def push_on_triangles(self, bt):
        aggregates = np.zeros((bt.mesh.num_triangles, 1))             
        missed     = defaultdict(list)
        triangles_hit = defaultdict(list) 
        for i, g in enumerate(self):
            vertex_id, twist_id = self.camera_position(i)
            y_coord, x_coord = np.where(g != 0)
            if not bt.is_legit_view_and_twist(vertex_id, twist_id):                
                raise ValueError('Back_Tracer and View_Gradients don\'t agree on the set of views.')                  
            for x,y in zip(x_coord, y_coord):
                try:
                    triangle = bt.from_2D_to_3D((x,y), vertex_id, twist_id)
                    aggregates[triangle] += g[y, x]
                    triangles_hit[(vertex_id, twist_id)].append((triangle))
                                                        
                except:              
                    missed[(vertex_id, twist_id)].append((x,y))                                        
        return aggregates, triangles_hit, missed   # TODO-Trim.
    
    def export_grads_to_txt(self, save_dir):
        '''
        Exports the grads attribute into .txt files. Each file corresponds to one view (vertex_id, twist_id)
        and lists every pixel where a gradient is positive into a separate line.
        '''
        nn_io.create_dir(save_dir)
        for i, grad in enumerate(self):
            y_coord, x_coord = np.where(grad)
            vertex_id, twist_id = self.camera_position(i)            
            out_file = 'grads_' + str(vertex_id) + '_' + str(twist_id) + '.txt'
            out_file = osp.join(save_dir, out_file)
            nn_io.write_pixel_list_to_txt(x_coord, y_coord, out_file)
     
if __name__ == '__main__':
    in_mesh = Mesh('../Data/Screw/screw.off')
    views   = Shape_Views('../Data/Screw/Views', 'png')
    grads   = View_Gradients(views, '../Data/Screw/raw_grads.npz')        
    rendered_size = (256, 256)
    grads.resize(rendered_size)
    grads.clean_grad_outside_shape_mask()
        
    bt      = Back_Tracer('../Data/Screw/Salient_Triangles', in_mesh)
    saliency_scores = np.zeros((in_mesh.num_triangles, 1))

    in_mesh.plot(triangle_function=saliency_scores)
