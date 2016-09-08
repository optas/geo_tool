'''
Created on Jul 15, 2016

@author:    Panos Achlioptas
@contact:   pachlioptas@gmail.com
@copyright: You are free to use, change, or redistribute this code in any
            way you want for non-commercial purposes. 
'''

import sys
import os.path as osp
import matplotlib.pyplot as plt


from nn_saliency.src.mesh import Mesh
from nn_saliency.src.laplace_beltrami import Laplace_Beltrami
from nn_saliency.src.back_tracer import Back_Tracer
import nn_saliency.src.nn_io as nn_io
import nn_saliency.src.mesh_cleaning as cleaning
from multiprocessing import Pool
from subprocess import call as sys_call

percent_of_eigs = 0.15
min_eigs = 7
max_eigs = 500
time_horizon = 20
min_vertices = 7

cmap = plt.get_cmap('jet')

global top_in_dir
global out_dir
fythumb_bin = '/home/panos/Renderer/a/b/fythumb_mvcnn/build/fythumb'


def render_views_with_fythumb(mesh_file, output_dir):
    sys_call([fythumb_bin, '-i', mesh_file, '-o', output_dir, '-r'])


def extract_hks_color(off_file):
    in_mesh = Mesh(off_file)
    in_mesh.center_in_unit_sphere()
    cleaning.clean_mesh(in_mesh, level=3, verbose=False)
    in_lb = Laplace_Beltrami(in_mesh)
    v_color = in_mesh.color_via_hks_of_component_spectra(in_lb, percent_of_eigs, time_horizon, min_vertices, min_eigs, max_eigs)
    v_color = in_mesh.adjacency_matrix().dot(v_color)
    v_color = cmap(v_color)
    out_file = off_file.replace(top_in_dir, out_dir)
    nn_io.write_off(out_file, in_mesh.vertices, in_mesh.triangles, vertex_color=v_color)


def extract_parts_color(off_file):
    in_mesh = Mesh(off_file)
    cleaning.clean_mesh(in_mesh, level=3, verbose=False)
    _, node_labels = in_mesh.connected_components()
    v_color = cmap(node_labels)
    out_file = off_file.replace(top_in_dir, out_dir)
    nn_io.write_off(out_file, in_mesh.vertices, in_mesh.triangles, vertex_color=v_color)
    views_out_dir = off_file.replace(top_in_dir, out_dir)[:-4]
    render_views_with_fythumb(out_file, views_out_dir)

if __name__ == '__main__':
    top_in_dir = osp.abspath(sys.argv[1])
    out_dir = osp.abspath(sys.argv[2])
    nn_io.copy_folder_structure(top_in_dir, out_dir)
    off_files = nn_io.files_in_subdirs(top_in_dir, '\.off$')

    pool = Pool(processes=6)
    for i, off_file in enumerate(off_files):
        print off_file
        pool.apply_async(extract_parts_color, args=(off_file,))
    pool.close()
    pool.join()
