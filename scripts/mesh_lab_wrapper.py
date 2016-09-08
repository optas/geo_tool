'''
Created on Jun 14, 2016

@author:    Panos Achlioptas
@contact:   pachlioptas@gmail.com
@copyright: You are free to use, change, or redistribute this code in any
            way you want for non-commercial purposes. 
'''

import sys
from subprocess import call as sys_call 
from os import path as osp

git_path = '/Users/t_achlp/Documents/Git_Repos/'
sys.path.insert(0, git_path)
import nn_saliency.src.nn_io as nn_io


mesh_lab_binary = '/Applications/meshlab.app/Contents/MacOS/meshlabserver'


def apply_script_to_meshes(top_folder, out_folder, script, regex='.+off$'):
    mesh_files = nn_io.files_in_subdirs(top_folder, regex)        
    nn_io.copy_folder_structure(top_folder, out_folder)
    for in_f in mesh_files:                         
        out_f = osp.join(out_folder, in_f.replace(top_folder, '')) 
        sys_call([mesh_lab_binary, '-i',  in_f, '-o',  out_f, '-s', script])                