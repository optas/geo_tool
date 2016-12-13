'''
Created on December 8, 2016

@author: Panayotes Achlioptas and Lin Shao
@contact: pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

import numpy as np
from .. utils import linalg_utils as utils
l2_norm = utils.l2_norm

class Bounding_Box(object):
    '''
    A class representing a 2D or 3D Bounding_Box.    
    '''

    def __init__(self, corners):
        '''
        Constructor.
        corners is a numpy array containing 4 or 6 non-negative integers.
        [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax]          
        '''
        self.corners = corners
        self.is_2d = len(self.corners) == 4
        self.is_3d = len(self.corners) == 6 
                                           
    def get_corners(self):
        ''' Syntactic sugar to get the corners property into separate variables.
        '''
        c = self.corners
        if self.is_2d:
            return c[0], c[1], c[2], c[3]
        else:
            return c[0], c[1], c[2], c[3], c[4], c[5]
        
    def area(self):
        if self.is_3d:
            raise Error('Bound Box is 3D thus it defines a Volume.')
        c = self.corners
        return (c[2] - c[0]) * (c[3] - c[1]) 
        
    def volume(self):
        if self.is_2d:
            raise ValueError('Bound Box is 2D thus it defines an area.')
        c = self.corners
        return (c[3] - c[0]) * (c[4] - c[1]) * (c[5] - c[2])
    
    def intersection_with(self, other):        
        if self.is_3d:
            [sxmin, symin, szmin, sxmax, symax, szmax] = self.get_corners()
            [oxmin, oymin, ozmin, oxmax, oymax, ozmax] = other.get_corners()
            dx = min(sxmax, oxmax) - max(sxmin, oxmin)
            dy = min(symax, oymax) - max(symin, oymin)
            dz = min(szmax, ozmax) - max(szmin, ozmin)
            inter = 0
	    
            if (dx > 0) and (dy > 0 ) and (dz > 0):                
                inter = dx * dy * dz
            
            return inter
                	
        else:                        
            [sxmin, symin, sxmax, symax]  = self.get_corners()
            [oxmin, oymin, oxmax, oymax]  = other.get_corners()
            dx = min(sxmax, oxmax) - max(sxin, oxmin)
            dy = min(symax, oymax) - max(symin, oymin)
            inter = 0
            if (dx > 0) and (dy > 0):
                inter = dx * dy 
            return inter        
        
    def diagonal_lengths(self):
        ''' Returns the length of the diagonals of a bounding box (bbox). If the bbox is 2D
        one length is returned. If bbox is 3D, the first returned value is the diagonal of the (x-y) 
        rectangle and the second the diagonal of the (x-y-z) bbox.           
        '''        
        if self.is_3d:
            [xmin, ymin, xmax, ymax, zmin, zmax]  = self.get_corners()
            two_d_diag = l2_norm([xmin-xmax, ymin-ymax])
            three_d_diag = l2_norm([xmin-xmax, ymin-ymax, zmin-zmax])
            return two_d_diag, three_d_diag
        else: 
            [xmin, ymin, xmax, ymax]  = self.get_corners()
            two_d_diag = l2_norm([xmin-xmax, ymin-ymax])            
            return two_d_diag
        
    def union_with(self, other):
        if self.is_3d:
            return self.volume()  + other.volume() - self.intersection_with(other)            
        else:
            return self.area() + other.area() - self.intersection_with(other)
    
    def iou_with(self, other):
        inter = self.intersection_with(other)
        union = self.union_with(other)
        return float(inte) / union 
    
    def overlap_ratio_with(self, other, ratio_type='union'):
        '''
        Returns the overlap ratio between two bounding boxes. That is the ratio of their (area or volume) intersection
        and their overlap. If the ratio_type is 'union' then the overlap is their (area/volume) of their union. If it is min, it 
        the min area/volume between them.  
        '''
        inter = self.intersection_with(other)
        if ratio_type == 'union':
            union = self.union_with(other)
            return float(inte) / union
        elif ratio_type == 'min':
            if self.is_2d:
                return float(inte) / min(self.area(),  other.area())
            else:
                return float(inte) / min(self.volume(),  other.volume())
        else:
            ValueError('ratio_type must be either \'union\', or \'min\'.')
    
    @staticmethod
    def bounding_box_of_3d_points(points):
        xmin = np.min(points[:,0])
        xmax = np.max(points[:,0])        
        ymin = np.min(points[:,1])
        ymax = np.max(points[:,1])
        zmin = np.min(points[:,2])                
        zmax = np.max(points[:,2])
        return Bounding_Box(np.array([xmin, ymin, zmin, xmax, ymax, zmax]))
