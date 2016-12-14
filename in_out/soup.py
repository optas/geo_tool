'''
Created on June 14, 2016

@author:    Panayotes Achlioptas
@contact:   pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

import cv2
import re
import os
import os.path as osp
import numpy as np
from glob import glob

from external_tools.python_plyfile.plyfile import PlyData

# TODO Break down in more in_out modules

def per_image_whitening(image):
    '''Linearly scales image to have zero mean and unit (variance) norm. The transformation is happening in_place.
    '''
    if image.dtype != np.float32:
        image = image.astype(np.float32, copy=False)

    if image.ndim == 2:
        n_elems = np.prod(image.shape)
        image -= np.mean(image)
        image /= max(np.std(image), 1.0 / np.sqrt(n_elems))        # Cap stdev away from zero.
    else:
        raise NotImplementedError

    assert(np.allclose([np.mean(image), np.var(image)], [0, 1], atol=1e-05, rtol=0))
    return image


def name_to_cam_position(file_name, cam_delim='-'):
    '''Given a filename produced by the FyThumb program, return the camera positions (vertex id) and
    rotation angle of the underlying rendered image file.
    '''
    file_name = os.path.basename(file_name)
    match = re.search(r'%s?(\d+)%s(\d+)' % (cam_delim, cam_delim), file_name)
    cam_index = int(match.group(1))
    rot_index = int(match.group(2))
    return [cam_index, rot_index]


def read_triangle_file(file_name):
    with open(file_name, 'r') as f_in:
        all_lines = f_in.readlines()
    n = len(all_lines)
    triangles  = np.empty((n, 3), dtype=np.int32)
    pixels     = np.empty((n, 2), dtype=np.int32)
    hit_coords = np.empty((n, 3), dtype=np.float32)
    for i in xrange(n):
        tokens          = all_lines[i].rstrip().split(' ')
        pixels[i,:]     = [int(tokens[0]), int(tokens[1])]
        triangles[i,:]  = [int(tokens[2]), int(tokens[3]), int(tokens[4])]
        hit_coords[i,:] = [float(tokens[5]), float(tokens[6]), float(tokens[7])]                     
    return pixels, triangles, hit_coords


def load_views_of_shape(view_folder, file_format, shape_views=None, reshape_to=None):
    view_list = []
    cam_pos = []
    view_mask = []

    searh_pattern = os.path.join(view_folder, '*.' + file_format)
    for view in glob(searh_pattern):
        image = cv2.imread(view, 0)     # Convert to Gray-Scale
        if reshape_to is not None:
            image = cv2.resize(image, reshape_to)

        reshape_to = image.shape
        view_mask.append(image != 0)  # Compute mask before whiten is applied.
        #if whiten:
        #   image = per_image_whitening(image)
        view_list.append(image)
        cam_pos.append(name_to_cam_position(view))

    if shape_views and len(view_list) != shape_views:
        raise IOError('Number of view files (%d) doesn\'t match the expected ones (%d)' % (len(view_list), shape_views))
    elif len(view_list) == 0: 
        raise IOError('There are no files of given format in this folder.')
    else:
        shape_views = len(view_list)

    views_tensor = np.reshape(view_list, (shape_views, reshape_to[0], reshape_to[1]))    
    return views_tensor, cam_pos, np.array(view_mask)


def format_image_tensor_for_tf(im_tensor, whiten=True):    
    new_tensor = np.zeros(im_tensor.shape, dtype=np.float32)
    if whiten:
        for ind, im in enumerate(im_tensor):
            new_tensor[ind,:,:] = per_image_whitening(im)            
    return np.expand_dims(new_tensor, 3)             # Add singleton trailing dimension.

def load_crude_point_cloud(file_name, delimiter=' ', comments='#'):
    '''
    Input: file_name (string) of a file containing 3D points. Each line of the file 
    is expected to contain exactly one point. The x,y,z coordinates of the point are separated via the provided 
    delimiter character(s).     
    '''
    return np.loadtxt(file_name, dtype=np.float32, comments=comments, delimiter=delimiter)

def load_annotation_of_points(file_name, format='shape_net'):
    '''
    Loads the annotation file that describes for every point of a point cloud which part it belongs too. 
    '''
    if format == 'shape_net':
        return np.loadtxt(file_name, dtype=np.int16)
    else:
        ValueError('NIY.')
                      

def load_ply(file_name, with_faces=False):
    ply_data = PlyData.read(file_name)
    points = ply_data['vertex']
    points = np.vstack([points['x'], points['y'], points['z']]).T
    
    if with_faces:
        faces = np.vstack(ply_data['face']['vertex_indices'])
        return points, faces
    else:
        return points

def load_off(file_name):
    _float = np.float32
    _int   = np.int32   
    break_floats = lambda in_file : [_float(s) for s in in_file.readline().strip().split(' ')]
    
    with open(file_name, 'r') as f_in:
        header = f_in.readline().strip()
        if header not in ['OFF', 'COFF']: 
            raise ValueError( 'Not a valid OFF header.')            
        
        n_verts, n_faces, _ = tuple([_int(s) for s in f_in.readline().strip().split(' ')]) # Disregard 3rd argument: n_edges.         
        
        verts = np.empty((n_verts, 3) , dtype=_float)
        v_color = None        
        first_line = break_floats(f_in)         
        verts[0,:] =  first_line[:3]                        
        if len(first_line) > 3:
            v_color = np.empty((n_verts, 4) , dtype=_float)            
            v_color[0,:] = first_line[3:]
            for i in xrange(1, n_verts):
                line = break_floats(f_in)
                verts[i,:] = line[:3] 
                v_color[i,:] = line[3:]
        else:            
            for i in xrange(1, n_verts):
                verts[i,:] = break_floats(f_in)
        
        first_line = [s for s in f_in.readline().strip().split(' ')]
        poly_type  = int(first_line[0])   # 3 for triangular mesh, 4 for quads etc.
        faces      = np.empty((n_faces, poly_type) , dtype=_int)
        faces[0:]  = [_int(f) for f in first_line[1:poly_type+1]] 
        f_color    = None        
        if len(first_line) > poly_type + 1:  # Color coded faces.          
            f_color = np.empty((n_faces, 4) , dtype=_float)
            f_color[0,:] = first_line[poly_type+1:]  
            for i in xrange(1, n_faces):
                line = [s for s in f_in.readline().strip().split(' ')]
                ptype = int(line[0])                
                if ptype != poly_type:
                    raise ValueError('Mesh contains faces of different dimensions. Loader in not yet implemented for this case.')
                faces[i,:] = [_int(f) for f in line[1:ptype+1]]
                f_color[i,:] =[_float(f) for f in line[ptype+1:]]                                 
        else:
            for i in xrange(1, n_faces):
                line =  [_int(s) for s in f_in.readline().strip().split(' ')]
                if line[0] != poly_type:
                    raise ValueError('Mesh contains faces of different dimensions. Loader in not yet implemented for this case.')                
                faces[i,:] = line[1:]
                
    if v_color is not None and f_color is not None: 
        return verts, faces, v_color, f_color
    if v_color is not None:
        return verts, faces, v_color
    if f_color is not None:
        return verts, faces, f_color
    return verts, faces
    
def write_off(out_file, vertices, faces, vertex_color=None, face_color=None):
    nv = len(vertices)
    nf,  tf = faces.shape
    if tf != 3:
        raise ValueError('Not Implemented Yet.') 
    
    vc = not(vertex_color is None)
    fc = not(face_color   is None)
    
    if vc and fc:
        raise ValueError('Color can be specified for the faces or the vertices - not both.')
                                       
    with open(out_file, 'w') as fout:
        if vc or fc:
            fout.write('COFF\n')
        else:
            fout.write('OFF\n')
            
        fout.write('%d %d 0\n' % (nv, nf))  # The third number is supposed to be the num of edges - but is set to 0 as per common practice. 
        
        if vc:
            c = vertex_color 
            for i, v in enumerate(vertices):
                fout.write('%f %f %f %f %f %f %f\n' % (v[0], v[1], v[2], c[i, 0], c[i, 1], c[i, 2], c[i, 3]))
            for f in faces:            
                fout.write('%d %d %d %d\n' % (tf, f[0], f[1], f[2]))
        elif fc:
            for v in vertices:
                fout.write('%f %f %f\n' % (v[0], v[1], v[2]))
            c = face_color 
            for i, f in enumerate(faces):
                fout.write('%d %d %d %d %f %f %f %f\n' % (tf, f[0], f[1], f[2], c[i, 0], c[i, 1], c[i, 2], c[i, 3] ))
        else:
            for v in vertices:
                fout.write('%f %f %f\n' % (v[0], v[1], v[2]))
            for f in faces:            
                fout.write('%d %d %d %d\n' % (tf, f[0], f[1], f[2]))      
                           
def write_pixel_list_to_txt(x_coord, y_coord, outfile):
    pixels = np.vstack([x_coord, y_coord]) 
    pixels = np.transpose(pixels)        # Write each pixel pair on each own line.
    np.savetxt(outfile, pixels, fmt='%d')

def read_pixel_list_from_txt(in_file):
    pixels  = np.loadtxt(in_file, dtype=np.int32)
    x_coord = pixels[:, 0]
    y_coord = pixels[:, 1]
    return x_coord, y_coord

def files_in_subdirs(top_dir, search_pattern):
    res = []    
    join = os.path.join
    regex = re.compile(search_pattern)
    for path, _, files in os.walk(top_dir):
        for name in files:
            full_name = join(path, name)
            if regex.search(full_name):
                res.append(full_name)
    return res

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
                
def copy_folder_structure(top_dir, out_dir):    
    if top_dir[-1] != os.sep:
        top_dir += os.sep
    
    all_dirs  = (dir_name for dir_name, _, _ in os.walk(top_dir))
    all_dirs.next() # Exhaust first name which is identical to the top_dir    
    for d in all_dirs:                
        create_dir(osp.join(out_dir, d.replace(top_dir, '')))

# ADD as way of making mask from image.  
# image         = cv2.imread(views_folder + '/img-00-00.png', cv2.IMREAD_UNCHANGED)
# plt.subplot(1,2,1); plt.imshow(image)
# test = np.asarray(image)
# xpix, ypix = test.shape[0:2]
# mask_test = np.zeros((xpix, ypix))
# white_rgb = np.array([0,0,0], dtype=test.dtype)
# for i in xrange(xpix):
#     for j in xrange(ypix):
#         if all(test[i,j,0:3] == white_rgb) and test[i,j,3] ==0:
#             mask_test[i,j] = 1
# plt.subplot(1,2,2); plt.imshow(mask_test)

