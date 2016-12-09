'''
Created on December 8, 2016

@author: Panos Achlioptas and Lin Shao
@contact: pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''


import copy
import numpy as np

from .. in_out import soup as io
from .. fundamentals.bounding_box import Bounding_Box

l2_norm = utils.l2_norm

 
class Point_Cloud(object): 
    '''
    A class representing a 3D Point Cloud.
    '''
    def __init__(self, point=None, ply_file=None):
        '''
        Constructor
        '''
        if off_file != None:            
            ply_data = PlyData.read(ply_file)
            points = ply_data['vertex']
            points = np.vstack([points['x'], points['y'], points['z']]).T        
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

    def bbox_diagonal_length(self):
        '''
            Returns the length of the longest line possible for which
            the end points are two vertices of the mesh.
        '''
        return l2_norm(np.min(self.points, axis=0) - np.max(self.points, axis=0))
        
    def bbox(self):
        xmin = np.min(self.points, axis = 0)
        ymin = np.min(self.points, axis = 1)
        zmin = np.max(self.points, axis = 2)
        xmax = np.max(self.points, axis = 0)
        ymax = np.max(self.points, axis = 1)
        zmax = np.max(self.points, axis = 2)
        return Bounding_Box(np.array([xmin, ymin, zmin, xmaz, ymax, zmax]))
            
    def center_in_unit_sphere(self):
        radius = 0.5 * self.bbox_diagonal_length()
        self.points /= radius
        barycenter = np.sum(self.points, axis=0) / self.num_points
        self.points -= barycenter
        return self