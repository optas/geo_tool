'''
Created on Jul 25, 2016

@author: Panos Achlioptas
@contact: pachlioptas@gmail.com
@copyright: You are free to use, change, or redistribute this code in any way
you want for non-commercial purposes.
'''

import glob
import os
import os.path as osp
from subprocess import call as sys_call

from .. in_out import soup as io


class Back_Tracer():
    '''
    A class providing the basic functionality for converting information
    regarding 2D rendered output of Fythumb into back into thr 3D space
    of the Mesh models.
    '''

    fythumb_bin = '/home/panos/Renderer/a/b/fythumb_mvcnn/build/fythumb'

    def __init__(self, triangle_folder, in_mesh):
        '''
        Constructor.
        '''
        self.mesh = in_mesh
        self.map = Back_Tracer.generate_pixels_to_triangles_map(triangle_folder, in_mesh)

    def from_2D_to_3D(self, pixels, vertex_id, twist_id):
        return self.map[vertex_id, twist_id][pixels]

    def is_legit_view_and_twist(self, vertex_id, twist_id):
        return (vertex_id, twist_id) in self.map

    @staticmethod
    def render_views_of_shapes(top_dir, output_dir, regex):
        io.copy_folder_structure(top_dir, output_dir)
        mesh_files = io.files_in_subdirs(top_dir, regex)
        if output_dir[-1] != os.sep:
            output_dir += os.sep

        for f in mesh_files:
            out_sub_folder = f.replace(top_dir, output_dir)
            mark = out_sub_folder[::-1].find('.')  # Find last occurrence of '.' to remove the ending (e.g., .txt)
            if mark > 0:
                out_sub_folder = out_sub_folder[:-mark - 1]
            Back_Tracer.fythumb_compute_views_of_shape(f, out_sub_folder)

    @staticmethod
    def fythumb_compute_views_of_shape(mesh_file, output_dir):
        sys_call([Back_Tracer.fythumb_bin, '-i', mesh_file, '-o', output_dir, '-r'])

    @staticmethod
    def pixels_to_triangles(pixel_file, off_file, camera_vertex, camera_twist, output_dir, out_file_name):
        sys_call([Back_Tracer.fythumb_bin, '-i', off_file, '-s', pixel_file, '-o', output_dir,
                  '-v', camera_vertex, '-t', camera_twist, '-f', out_file_name])

    @staticmethod
    def compute_triangles_from_pixels(off_file, pixels_folder, output_folder):
        searh_pattern = osp.join(pixels_folder, '*.txt')
        c = 0
        for pixel_file in glob.glob(searh_pattern):
            camera_vertex, camera_twist = io.name_to_cam_position(pixel_file, cam_delim='_')
            out_file_name = '%d_%d.txt' % (camera_vertex, camera_twist)
            print 'Computing Triangles for %s file.' % (pixel_file)
            Back_Tracer.pixels_to_triangles(pixel_file, off_file, str(camera_vertex), str(camera_twist), output_folder, out_file_name)
            c += 1
        print 'Computed the triangles for %d files.' % (c)

    @staticmethod
    def generate_pixels_to_triangles_map(triangle_folder, in_mesh):
        searh_pattern = osp.join(triangle_folder, '*.txt')
        inv_dict = in_mesh.inverse_triangle_dictionary()
        res = dict()
        for triangle_file in glob.glob(searh_pattern):
            camera_vertex, camera_twist = io.name_to_cam_position(triangle_file, cam_delim='_')
            res[(camera_vertex, camera_twist)] = dict()
            pixels, triangles, _ = io.read_triangle_file(triangle_file)
            triangles = map(tuple, triangles)
            triangles = [inv_dict[tr] for tr in triangles]
            pixels = map(tuple, pixels)
            res[(camera_vertex, camera_twist)] = {key: val for key, val in zip(pixels, triangles)}
        return res

if __name__ == '__main__':    
    from geo_tool.solids.mesh import Mesh
    in_mesh = Mesh('../Data/Screw/screw.off')
    bt = Back_Tracer('../Data/Screw/Salient_Triangles', in_mesh)
    print bt
