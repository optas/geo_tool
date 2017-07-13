import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from subprocess import call

from geo_tool.in_out.soup import load_wavefront_obj, load_ply
from geo_tool import Point_Cloud
from geo_tool.external_code.python_plyfile.plyfile import PlyData, PlyElement

from general_tools.strings import trim_content_after_last_dot
from general_tools.in_out import create_dir

import shutil
import os.path as osp



class Mitsuba_Rendering(object):

    def __init__(self, command_file, model_list, img_out_dir, temp_dir, dependencies_dir, clean_temp=False):
        ''' Initializer of Class instance.
        Input:
            command_file : file_name where the commands calling Mitsuba will be written.
            model_list: a list containing the file_names of the models that will be rendered.
            img_out_dir: where the rendered files will be saved.
            temp_dir : Mitsuba will store here intermediate results.
            dependencies_dir: holds files like: envmap.exr, matpreview.serialized that are necessary for rendering.
            clean_temp: if True, post rendering all intermediate results will be deleted.
        '''

        self.command_file = command_file
        self.img_out_dir = img_out_dir
        self.temp_dir = temp_dir        
        self.model_list = model_list     
        self.clean_temp = clean_temp   
        self.set_default_rendering_params()
        
        create_dir(self.temp_dir)
        create_dir(self.img_out_dir)
        shutil.copy(os.path.join(dependencies_dir, 'envmap.exr'), self.temp_dir)
        shutil.copy(os.path.join(dependencies_dir, 'matpreview.serialized'), self.temp_dir) 
          
    def set_default_rendering_params(self):
        self.radius = 0.01          		# Size of rendered sphere (of point-clouds).
        self.ldsampler_n_samples = 128		# Controls the quality of the rendering, higher is better.
        self.zrot = (-50) * np.pi / 180;
        self.ytrans = 0.5
        self.ztrans = 0
        self.dx_scale = 1
        self.sensor_focus_distance = 2.3173
        self.sensor_target = [0, 0, 0]
        self.sensor_origin = [0, -1.25, 0.40]
        self.sensor_up = [0, 0, 1]
        self.sensor_height = 480
        self.sensor_width = 480
        self.scale = 10
        self.color = None
        
    def pc_loader(self, file_name, normalize=True):
        if file_name[-4:] == '.ply': 
            pc = Point_Cloud(ply_file=file_name)            
            if normalize:
                pc.center_in_unit_sphere()
                
            pc.rotate_z_axis_by_degrees(self.zrot, clockwise=False)
        return pc

#     def obj_loader(self, file_name):
#         vertices, faces, normals = load_wavefront_obj(file_name)
#         v_np = np.zeros((len(vertices),3))
#         for i in xrange(len(vertices)):
#             v_np[i,:] = vertices[i] 
#         return v_np, faces 
    
    def generate_commands_for_point_cloud_rendering(self):             
        command = ''
        for model_file in self.model_list:
            print model_file                 
            pcloud = self.pc_loader(model_file)
            pc_z_min = np.min(pcloud.points[:, 2])
            
            model_name = trim_content_after_last_dot(osp.basename(model_file))
            xml_file = os.path.join(self.temp_dir, model_name + '.xml')
            
            with open(xml_file, 'w') as xml_out:
                xml_out.write(self.xml_string(pc_z_min))
                for point in pcloud.points:                
                    xml_out.write(self.xml_point_string(self.radius, point))                
                xml_out.write(self.xml_closure())
        
            img_file = os.path.join(self.img_out_dir, model_name + '.png')
            exr_file = os.path.join(self.temp_dir, model_name + '.exr')
            
            open(exr_file, 'w').close() # create empty file                
        
            command += 'mitsuba %s\n' % xml_file
            command += 'mtsutil tonemap -o %s %s\n\n' % (img_file, exr_file)
        
                            
        with open(self.command_file, 'w') as fout:
            fout.write(command)
            if self.clean_temp:
                command = 'rm mitsuba*.log\n'
                command += 'rm -rf %s\n' % (self.temp_dir, )
                fout.write(command)
        
        try:
            os.system("chmod +x %s" % (self.command_file, ))        
        except:
            pass
            
    def xml_string(self,pc_z_min):
        return self.xml_preamble() + self.xml_sensor() + self.xml_emitter() + self.xml_fold(pc_z_min) 
    
    def xml_point_string(self, radius, position, color=None):
        if color is not None:
            Blues = plt.get_cmap('autumn')
            r, g, b, _  = Blues(- int(color * 400) + 124)
            color_value = "%f %f %f" % (r, g, b)
        else:
            color_value = '#6d7185'
 
        out_str = '<shape type="sphere">\n'
        out_str += '\t<float name="radius" value="%.5f"/>\n' % radius
        out_str += '\t<transform name="toWorld">\n'
        out_str += '\t\t<translate x="%.10f" y="%.10f" z="%.10f"/>\n' % (position[0], position[1], position[2])
        out_str += '\t</transform>\n'
        out_str += '\t<bsdf type="diffuse">\n'
        out_str += '\t\t<srgb name="diffuseReflectance" value="%s"/>\n' % (color_value, )        
        # out_str += '\t\t<float name="intIOR" value="1.9"/>\n'  # TODO Olga do we need this?
        out_str += '\t</bsdf>\n'  
        out_str += '</shape>\n\n'
         
        return out_str
 
    def xml_preamble(self):
        out_str = '<?xml version="1.0" encoding="utf-8"?>\n\n'  
        out_str += '<scene version="0.5.0">\n' 
        out_str += '\t<!--Setup scene integrator -->\n'  
        out_str += '\t<integrator type="path">\n' 
        out_str += '\t\t<!-- Path trace with a max. path length of 5 -->\n' 
        out_str += '\t\t<integer name="maxDepth" value="5"/>\n' 
        out_str += '\t</integrator>\n\n'
        return out_str
 
    def xml_sensor(self):
        out_str = '<sensor type="perspective">\n'
        out_str += '\t<float name="focusDistance" value="%.10f"/>\n' % self.sensor_focus_distance
        out_str += '\t<float name="fov" value="45"/>\n'
        out_str += '\t<string name="fovAxis" value="x"/>\n'
        out_str += '\t<transform name="toWorld">\n'
        out_str += '\t\t<lookat target="%.10f, %.10f, %.10f" origin="%.10f, %.10f, %.10f" up="%.10f, %.10f, %.10f"/>\n' % (self.sensor_target[0], self.sensor_target[1], self.sensor_target[2], self.sensor_origin[0], self.sensor_origin[1], self.sensor_origin[2], self.sensor_up[0], self.sensor_up[1], self.sensor_up[2])
        out_str += '\t</transform>\n'
        out_str += '\t<sampler type="ldsampler">\n'
        out_str += '\t\t<integer name="sampleCount" value="%d"/>\n' % self.ldsampler_n_samples
        out_str += '\t</sampler>\n'
        out_str += '\t<film type="hdrfilm">\n'
        out_str += '\t\t<integer name="height" value="%i"/>\n' % self.sensor_height
        out_str += '\t\t<integer name="width" value="%i"/>\n' % self.sensor_width
        out_str += '\t\t<rfilter type="gaussian"/>\n'
        out_str += '\t</film>\n'
        out_str += '</sensor>\n\n'
        return out_str
    
    def xml_emitter(self):
        out_str =  '<emitter type="envmap" id="Area_002-light">\n'
        out_str += '\t<string name="filename" value="envmap.exr"/>\n'
        out_str += '\t<transform name="toWorld">\n'
        out_str += '\t\t<rotate y="1" angle="-180"/>\n'
        out_str += '\t\t<matrix value="-0.224951 -0.000001 -0.974370 0.000000 -0.974370 0.000000 0.224951 0.000000 0.000000 1.000000 -0.000001 8.870000 0.000000 0.000000 0.000000 1.000000"/>\n'
        out_str += '\t</transform>\n'
        out_str += '\t<float name="scale" value="3"/>\n'
        out_str += '</emitter>\n\n'
        return out_str
  
    def xml_obj(self, obj_filename, position=[0, 0, 0]):
        out_str =  '<shape type="obj">\n'
        out_str += '\t<string name="filename" value="%s"/>\n' % obj_filename
        out_str += '\t<transform name="toWorld" >\n'
        out_str += '\t\t<translate x="%.10f" y="%.10f" z="%.10f"/>\n' % (position[0], position[1], position[2])
        out_str += '\t</transform >\n'
        out_str += '\t<bsdf type="diffuse" >\n'
        out_str += '\t\t<srgb name="diffuseReflectance" value="#6d7185"/>\n'
        # out_str += '\t\t<float name="intIOR" value="1.9"/>\n' #TODO olga?
        out_str += '\t</bsdf>\n'
        out_str += '</shape>\n'
        return out_str
 
    def xml_fold(self, pc_z_min=-0.5):        
        out_str = '<texture type="checkerboard" id="__planetex">\n'        
        out_str += '\t<rgb name="color0" value="0.9"/>\n'
        out_str += '\t<rgb name="color1" value="0.9"/>\n'
        out_str += '\t<float name="uscale" value="8.0"/>\n'
        out_str += '\t<float name="vscale" value="8.0"/>\n'
        out_str += '\t<float name="uoffset" value="0.0"/>\n'
        out_str += '\t<float name="voffset" value="0.0"/>\n'        
        out_str += '</texture>\n\n'
 
        out_str += '<bsdf type="diffuse" id="__planemat">\n'
        out_str += '\t<ref name="reflectance" id="__planetex"/>\n'
        out_str += '</bsdf>\n\n'
 
        out_str += '<shape type="serialized" id="Plane-mesh_0">\n'
        out_str += '\t<string name="filename" value="matpreview.serialized"/>\n'
        out_str += '\t<integer name="shapeIndex" value="0"/>\n'
        out_str += '\t<transform name="toWorld">\n'
        out_str += '\t\t<scale x="%.4f" y="%.4f" z="%.4f"/>\n' % (self.scale, self.scale, self.scale)
        out_str += '\t\t<translate y="%.10f" z="%.10f" />\n' % (self.scale * self.ytrans, self.ztrans + pc_z_min)
        out_str += '\t</transform>\n'
        out_str += '\t<ref name="bsdf" id="__planemat"/>\n'
        out_str += '</shape>\n\n'
        return out_str
  
    def xml_closure(self):
        return '</scene>\n'
 
     
#     def obj_commands(self):
#         self.num_commands = len(self.model_list)
#         fw_command = open(os.path.join(self.command_dir, 'command.sh'), 'w')
#         for i in xrange(self.num_commands):
#             obj_filename = os.path.join(self.model_list, self.model_list[i] + self.file_extension)
#             points, faces = self.obj_loader(obj_filename)
#             points = points.dot(self.rotation_mat().transpose())  
#             xml_path = os.path.join(self.temp_dir, self.model_list[i]+'.xml')
#             exr_path = os.path.join(self.temp_dir, self.model_list[i]+'.exr')
#             img_path = os.path.join(self.img_out_dir, self.model_list[i]+'.png')
#             fw_exr = open(exr_path,'w')
#             fw_exr.close() 
#             fw_xml = open(xml_path, 'w')
#             fw_xml.write(self.xml_string(-0.5))
#             fw_xml.write(self.xml_obj(obj_filename)) 
#             fw_xml.write(self.xml_post())     
#             fw_xml.close()
#             command = 'mitsuba %s\n' % xml_path
#             command += 'mtsutil tonemap -o %s %s\n' % (img_path, exr_path)
#             fw_command.write(command)
#         fw_command.close()