'''
Created on December 8, 2016

@author: Panos Achlioptas and Lin Shao
@contact: pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''


import copy
import numpy as np

from external_tools.python_plyfile.plyfile import PlyData

from .. in_out import soup as io
from .. utils import linalg_utils as utils
from .. fundamentals.bounding_box import Bounding_Box


class Point_Cloud(object): 
    '''
    A class representing a 3D Point Cloud.
    Dependencies:
        1. plyfile 0.4: PLY file reader/writer. DOI: https://pypi.python.org/pypi/plyfile
    '''
    def __init__(self, point=None, ply_file=None):
        '''
        Constructor
        '''
        if ply_file != None:            
            ply_data = PlyData.read(ply_file)
            points = ply_data['vertex']
            self.points = np.vstack([points['x'], points['y'], points['z']]).T        
        else:
            self.point = points 
                        
    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, value):
        self._points  = value
        self.num_points = len(self._points)

    def __str__(self):
        return 'Point Cloud with %d points.' % (self.num_points)

    def copy(self):
        return copy.deepcopy(self)

    def bounding_box(self):
        xmin = np.min(self.points[:,0])
        xmax = np.max(self.points[:,0])        
        ymin = np.min(self.points[:,1])
        ymax = np.max(self.points[:,1])
        zmin = np.min(self.points[:,2])                
        zmax = np.max(self.points[:,2])
        return Bounding_Box(np.array([xmin, ymin, zmin, xmax, ymax, zmax]))
            
    def center_in_unit_sphere(self):
        radius = 0.5 * self.bbox_diagonal_length()
        self.points /= radius
        barycenter = np.sum(self.points, axis=0) / self.num_points
        self.points -= barycenter
        return self