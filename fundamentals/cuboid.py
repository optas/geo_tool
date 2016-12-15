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
    
    def is_point_inside(self, point):
        '''Given a 3D point tests if it lies inside the Cuboid.
        '''
        [xmin, ymin, zmin, xmax, ymax, zmax] = self.extrema
        return np.all([xmin, ymin, zmin]<=point) and np.all([xmax, ymax, zmax]>=point) 
                     
    def containing_sector(self, sector_center, ignore_z_axis=True):
        '''Computes the tightest (conic) sector that contains the Cuboid. The sector's center is defined by the user.
        Input:
            sector_center: 3D Point where the sector begins.
            ignore_z_axis: (Boolean) if True the Cuboid is treated as rectangle by eliminating it's z-dimension.
        Notes: Roughly it computes the angle between the ray's starting at the sector's center and each side of the cuboid. 
        The one with the largest angle is the requested sector.          
        '''        
        if self.is_point_inside(sector_center):
            raise ValueError('Sector\'s center lies inside the bounding box.')
            
        def angle_of_sector(sector_center, side):
            x1, y1, x2, y2 = side
            line_1 = np.array([x1 - sector_center[0], y1 - sector_center[1]])  # First diagonal pair of points of cuboid   
            line_2 = np.array([x2 - sector_center[0], y2 - sector_center[1]])
            cos =  line_1.dot(line_2) / (l2_norm(line_1) * l2_norm(line_2))
            if cos >= 1 or cos <= -1:
                angle = 0
            else:              
                angle = np.arccos(cos)
                assert(angle <= np.pi and angle >= 0)
            return angle        
        
        if ignore_z_axis:                                            
            [xmin, ymin, _, xmax, ymax, _] = self.extrema
            sides = [ [xmin, ymin, xmax, ymax],                                                        
                      [xmax, ymin, xmin, ymax],                      
                      [xmin, ymax, xmax, ymax],                                            
                      [xmin, ymin, xmax, ymin],                      
                      [xmin, ymin, xmin, ymax],
                      [xmax, ymin, xmax, ymax],                                            
                    ]
            
            a0 = angle_of_sector(sector_center, sides[0])
            a1 = angle_of_sector(sector_center, sides[1])  # a0, a1: checking the diagonals.            
            a2 = angle_of_sector(sector_center, sides[2])
            a3 = angle_of_sector(sector_center, sides[3])
            a4 = angle_of_sector(sector_center, sides[4])
            a5 = angle_of_sector(sector_center, sides[5])            
            largest = np.argmax([a0, a1, a2, a3, a4, a5])
            return  np.array(sides[largest][0:2]), np.array(sides[largest][2:])
                                                           
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
    
    def plot(self, axis=None, c='r'):
        '''Plot the Cuboid.
        Input:
            axis - (matplotlib.axes.Axes) where the cuboid will be drawn.
            c - (String) specifying the color of the cuboid. Must be valid for matplotlib.pylab.plot
        '''
        corners = self.corner_points()
        if axis != None:
            axis.plot([corners[0,0], corners[1,0]], [corners[0,1], corners[1,1]], zs=[corners[0,2], corners[1,2]], c=c)
            axis.plot([corners[1,0], corners[2,0]], [corners[1,1], corners[2,1]], zs=[corners[1,2], corners[2,2]], c=c)
            axis.plot([corners[2,0], corners[3,0]], [corners[2,1], corners[3,1]], zs=[corners[2,2], corners[3,2]], c=c)
            axis.plot([corners[3,0], corners[0,0]], [corners[3,1], corners[0,1]], zs=[corners[3,2], corners[0,2]], c=c)
            axis.plot([corners[4,0], corners[5,0]], [corners[4,1], corners[5,1]], zs=[corners[4,2], corners[5,2]], c=c)
            axis.plot([corners[5,0], corners[6,0]], [corners[5,1], corners[6,1]], zs=[corners[5,2], corners[6,2]], c=c)
            axis.plot([corners[6,0], corners[7,0]], [corners[6,1], corners[7,1]], zs=[corners[6,2], corners[7,2]], c=c)
            axis.plot([corners[7,0], corners[4,0]], [corners[7,1], corners[0,1]], zs=[corners[7,2], corners[4,2]], c=c)
            axis.plot([corners[0,0], corners[4,0]], [corners[0,1], corners[4,1]], zs=[corners[0,2], corners[4,2]], c=c)
            axis.plot([corners[1,0], corners[5,0]], [corners[1,1], corners[5,1]], zs=[corners[1,2], corners[5,2]], c=c)
            axis.plot([corners[2,0], corners[6,0]], [corners[2,1], corners[6,1]], zs=[corners[2,2], corners[6,2]], c=c)
            axis.plot([corners[3,0], corners[7,0]], [corners[3,1], corners[7,1]], zs=[corners[3,2], corners[7,2]], c=c)
        else:
            ValueError('NYI')
                    
                    
    @staticmethod
    def bounding_box_of_3d_points(points):
        xmin = np.min(points[:,0])
        xmax = np.max(points[:,0])        
        ymin = np.min(points[:,1])
        ymax = np.max(points[:,1])
        zmin = np.min(points[:,2])                
        zmax = np.max(points[:,2])
        return Cuboid(np.array([xmin, ymin, zmin, xmax, ymax, zmax]))
