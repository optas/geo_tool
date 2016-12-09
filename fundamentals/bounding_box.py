'''
Created on December 8, 2016

@author: Panayotes Achlioptas and Lin Shao
@contact: pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

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
        c = self.corners
        if self.is_2d:
            return c[0], c[1], c[2], c[3]
        else:
            return c[0], c[1], c[2], c[3], c[4], c[5]
        
    def area(self):
        if not self.is_2d:
            raise Error('Bound Box is 3D thus it defines a Volume.')
        c = self.corners()
        return (c[2] - c[0]) * (c[3] - c[1]) 
        
    def volume(self):
        if not self.is_2d:
            raise Error('Bound Box is 3D thus it defines a Volume.')
        c = self.corners()
        return (c[3] - c[0]) * (c[4] - c[1]) * (c[5] - c[2])
    
    def intersection_with(self, other):        
        if self.is_3d:
            pass # TODO Lin Fill Here
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
        '''
            Returns the length of the longest line possible for which
            the end points are two points of the Point Cloud. 
        '''        
        if self.is_3d:
            [xmin, ymin, xmax, ymax, zmin, zmax]  = self.get_corners()
            two_d_diag = l2_norm([ymin-xmin, ymax-xmax])
            three_d_diag = l2_norm([xmin-xmax, ymin-ymax, zmin-zmax])
            return two_d_diag, three_d_diag
        else: 
            pass
    
    def union_with(self, other):
        if self.is_3d:
            sv = self.volume()
            ov = other.volume()
            return sv  + ov - self.intersection_with(other)            
        else:
            pass
    
    def iou_with(self, other):
        inter = self.intersection_with(other)
        union = self.union_with(other)
        return float(inte) / union 