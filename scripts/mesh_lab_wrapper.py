'''Created on June 14, 2016

@author:    Panos Achlioptas
@contact:   pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any
            way you want for non-commercial purposes.
'''


from subprocess import call as sys_call
from os import path as osp
from general_tools.in_out.basics import files_in_subdirs, copy_folder_structure

mesh_lab_binary = '/Applications/meshlab.app/Contents/MacOS/meshlabserver'


def apply_script_to_files(top_folder, out_top_dir, script_file, regex='.+off$'):
    input_files = files_in_subdirs(top_folder, regex)
    copy_folder_structure(top_folder, out_top_dir)
    for in_f in input_files:
        out_f = osp.join(out_top_dir, in_f.replace(top_folder, ''))
        sys_call([mesh_lab_binary, '-i', in_f, '-o', out_f, '-s', script_file])
