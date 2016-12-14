'''
Created on December 8, 2016

@author: Panayotes Achlioptas and Lin Shao
@contact: pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

import numpy as np
from .. utils import linalg_utils as utils
l2_norm = utils.l2_norm

class Cuboid(object):
    '''
    A class representing a 3D Cuboid.    
    '''

    def __init__(self, extrema):
        '''
        Constructor.
        extrema is a numpy array containing 6 non-negative integers [xmin, ymin, zmin, xmax, ymax, zmax].                
        '''
        self.extrema = extrema                
                                           
    def get_extrema(self):
        ''' Syntactic sugar to get the extrema property into separate variables.
        '''
        c = self.corners
        return c[0], c[1], c[2], c[3], c[4], c[5]
            
    def volume(self):    
        c = self.corners
        return (c[3] - c[0]) * (c[4] - c[1]) * (c[5] - c[2])
    
    def intersection_with(self, other):            
        [sxmin, symin, szmin, sxmax, symax, szmax] = self.get_extrema()
        [oxmin, oymin, ozmin, oxmax, oymax, ozmax] = other.get_extrema()
        dx = min(sxmax, oxmax) - max(sxmin, oxmin)
        dy = min(symax, oymax) - max(symin, oymin)
        dz = min(szmax, ozmax) - max(szmin, ozmin)
        inter = 0
    
        if (dx > 0) and (dy > 0 ) and (dz > 0):                
            inter = dx * dy * dz
        
        return inter
                	                
    def corner_points(self):
        [xmin, ymin, zmin, xmax, ymax, zmax] = self.extrema
        c1 = np.array([xmin, ymin, zmin])
        c2 = np.array([xmax, ymin, zmin])
        c3 = np.array([xmax, ymax, zmin])
        c4 = np.array([xmin, ymax, zmin])        
        c5 = np.array([xmin, ymin, zmax])
        c6 = np.array([xmax, ymin, zmax])
        c7 = np.array([xmax, ymax, zmax])
        c8 = np.array([xmin, ymax, zmax])
        return np.vstack([c1, c2, c3, c4, c5, c6, c7, c8])
                                                
    def barycenter(self):
        corners = self.corner_points()
        n_corners = corners.shape[0]
        return np.sum(corners, axis=0) / n_corners        
                                    
    def faces(self):
        corners = self.corner_points()
        [xmin, ymin, zmin, xmax, ymax, zmax] = self.extrema
        xmin_f = corners[corners[:,0] == xmin, :]
        xmax_f = corners[corners[:,0] == xmax, :]
        ymin_f = corners[corners[:,1] == ymin, :]
        ymax_f = corners[corners[:,1] == ymax, :]
        zmin_f = corners[corners[:,2] == zmin, :]
        zmax_f = corners[corners[:,2] == zmax, :]
        return [xmin_f, xmax_f, ymin_f, ymax_f, zmin_f, zmax_f]
                 
    def containing_sector(self, sector_center, ignore_z_axis=True):
        # TODO - break into multiple move to Point Clouds
        # TODO - check that it is feasible
        def angle_of_sector(sector_center, x1, y1, x2, y2):
            line_1 = np.array([x1 - sector_center[0], y1 - sector_center[1]])  # First diagonal pair of points of cuboid   
            line_2 = np.array([x2 - sector_center[0], y2 - sector_center[1]])
            ns1 = np.square(l2_norm(line_1))
            ns2 = np.square(l2_norm(line_2))        
            ns3 = np.square(l2_norm(np.array([x1-y1, x2-y2]) ))            
            cos1 = (ns1 + ns2 - ns3) / 2 * line_1.dot(line_2)
            a1 = np.arccos(cos1)
            assert(a1<=180 and a1>=0)
            return a1        
        
        if ignore_z_axis:
            [xmin, ymin, _, xmax, ymax, _] = self.extrema
            a1 = angle_of_sector(sector_center, xmin, ymin, xmax, ymax)
            a2 = angle_of_sector(sector_center, xmax, ymin, xmin, ymax)
            if a1>= a2:
                return np.array([xmin, ymin]), np.array([xmax, ymax])
            else:
                return np.array([xmax, ymin]), np.array([xmin, ymax])
                                                           
    def union_with(self, other):
        return self.volume()  + other.volume() - self.intersection_with(other)            
            
    def iou_with(self, other):
        inter = self.intersection_with(other)
        union = self.union_with(other)
        return float(inte) / union 
    
    def overlap_ratio_with(self, other, ratio_type='union'):
        '''
        Returns the overlap ratio between two cuboids. That is the ratio of their volume intersection
        and their overlap. If the ratio_type is 'union' then the overlap is the volume of their union. If it is min, it 
        the min volume between them.  
        '''
        inter = self.intersection_with(other)
        if ratio_type == 'union':
            union = self.union_with(other)
            return float(inte) / union
        elif ratio_type == 'min':            
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
        return Cuboid(np.array([xmin, ymin, zmin, xmax, ymax, zmax]))
