'''
Created on December 8, 2016

@author: Panayotes Achlioptas and Lin Shao
@contact: pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''


import copy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
    
from .. in_out import soup as io
from .. utils import linalg_utils as utils
from .. fundamentals import Cuboid

l2_norm = utils.l2_norm

class Point_Cloud(object): 
    '''
    A class representing a 3D Point Cloud.
    Dependencies:
        1. plyfile 0.4: PLY file reader/writer. DOI: https://pypi.python.org/pypi/plyfile
    '''
    def __init__(self, points=None, ply_file=None):
        '''
        Constructor
        '''
        if ply_file != None:            
            self.points = io.load_ply(ply_file)         
        else:
            self.points = points 
                        
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
        return Cuboid.bounding_box_of_3d_points(self.points)
    
    def center_in_unit_sphere(self):
        self.points = Point_Cloud.center_points_in_unit_sphere(self.points)
        return self
    
    def plot(self, *args, **kwargs):
        x = self.points[:,0]
        y = self.points[:,1]
        z = self.points[:,2]
        Point_Cloud.plot_3d_point_cloud(x, y, z, *args, **kwargs)
    
    def barycenter(self):
        n_points = self.points.shape[0]
        return np.sum(self.points, axis=0) / n_points
        
    @staticmethod
    def center_points_in_unit_sphere(points):
        n_points = points.shape[0]
        barycenter = np.sum(points, axis=0) / n_points
        points -= barycenter   # Center it in the origin.
        max_dist = np.max(l2_norm(points, axis=1)) # Make max distance equal to one.
        points /= max_dist * 2
        return points
    
    @staticmethod
    def plot_3d_point_cloud(x, y, z, *args, **kwargs):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, *args, **kwargs)
        plt.show()