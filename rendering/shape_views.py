'''
Created on Jul 25, 2016

@author: Panos Achlioptas
@contact: pachlioptas@gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

import numpy as np
import os.path as osp
import matplotlib.pylab as plt

import nn_io


class Shape_Views():
    '''
    classdocs
    '''

    def __init__(self, view_folder, file_format):
        '''
        Constructor
        '''
        data = nn_io.load_views_of_shape(view_folder, file_format)
        self.views = data[0]
        self.cam_pos = data[1]
        self.masks = data[2]
        self.inv_dict = {(pos[0], pos[1]): i for i, pos in enumerate(self.cam_pos)}

    def num_views(self):
        return self.views.shape[0]

    def image_size(self):
        return self.views[0].shape

    def plot(self, vertex_id, twist_id, mask=False):
        index = self.inv_dict[(vertex_id, twist_id)]
        if mask:
            im = self.masks[index, :, :]
        else:
            im = self.views[index, :, :]
#         plt.figure() # TO DO - keep opening new figures
        plt.imshow(im)
        plt.show()

    def export_masks_to_txt(self, save_dir):
        '''
        Exports the masks attribute into .txt files. Each file corresponds to one view (vertex_id, twist_id)
        and lists every pixel that belongs in the mask into a separate line.
        '''
        nn_io.create_dir(save_dir)
        for i, mask in enumerate(self.masks):
            y_coord, x_coord = np.where(mask)
            out_file = 'pixels_' + str(self.cam_pos[i][0]) + '_' + str(self.cam_pos[i][1]) + '.txt'
            out_file = osp.join(save_dir, out_file)
            nn_io.write_pixel_list_to_txt(x_coord, y_coord, out_file)

    def paint_masks_to_triangle_color(self, bt, tr_color):
        res = np.zeros(self.masks.shape)
        for i, mask in enumerate(self.masks):
            vertex_id, twist_id = self.cam_pos[i]
            if not bt.is_legit_view_and_twist(vertex_id, twist_id):                
                raise ValueError('Back_Tracer and View_Gradients don\'t agree on the set of views.')            
            y_coord, x_coord = np.where(mask != 0)
            for x, y in zip(x_coord, y_coord):
                try:
                    triangle = bt.from_2D_to_3D((x,y), vertex_id, twist_id)
                    res[i,y,x] = tr_color[triangle]
                except:
                    pass
        return res

if __name__ == '__main__':
    sv = Shape_Views('../Data/Screw/Views', 'png')
    sv.export_masks_to_txt('../Data/Screw/Masks')