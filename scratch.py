'''
Created on Jun 14, 2016

@author:    Panos Achlioptas
@contact:   pachlioptas@gmail.com
@copyright: You are free to use, change, or redistribute this code in any
            way you want for non-commercial purposes. 
'''


# TODO: check ravel() vs. np.repeat.
from solids.mesh import Mesh
import solids.mesh_cleaning as mc 
import os.path as osp


DATA_DIR = '/Users/optas/Documents/DATA/Model_Net_10'
class_type = 'bath_tub'
model_id = 0001
# off_file = osp.join(DATA_DIR, class_type, 'train', model_id, '.off')

off_file = '/Users/optas/Documents/DATA/Model_Net_10/OFF_Original/bathtub/train/bathtub_0001.off'

in_mesh = Mesh(off_file)
mc.clean_mesh(in_mesh, level=3, verbose=True)
in_mesh.plot()


