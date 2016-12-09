'''
Created on December 8, 2016

@author: Panos Achlioptas and Lin Shao
@contact: pachlioptas@gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

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
                                                             
    def is_2d(self):
        return len(self.corners) == 4 
    
    def is_3d(self):
        return len(self.corners) == 6
    
    def get_corners(self):
        c = self.corners
        if self.is_2d:
            return c[0], c[1], c[2], c[3]
        else:
            return c[0], c[1], c[2], c[3], c[4], c[5]
        
    def area(self):
        if not self.is_2d():
            raise Error('Bound Box is 3D thus it defines a Volume.')
        c = self.corners()
        return (c[2] - c[0]) * (c[3] - c[1]) 
        
    def volume(self):
        if not self.is_2d():
            raise Error('Bound Box is 3D thus it defines a Volume.')
        c = self.corners()
        return (c[3] - c[0]) * (c[4] - c[1]) * (c[5] - c[2])
    
    def intersection_with(self, other):
        [sxmin, symin, sxmax, symax]  = self.get_corners()
        [oxmin, oymin, oxmax, oymax]  = other.get_corners()
        if self.is_3d():
            pass # TODO Lin Fill Here
        else:                        
            dx = min(sxmax, oxmax) - max(sxin, oxmin)
            dy = min(symax, oymax) - max(symin, oymin)
            inter = 0
            if (dx > 0) and (dy > 0):
                inter = dx * dy 
            return inter             
    
    def union_with(self, other):
        if self.is_3d():
            sv = self.volume()
            ov = other.volume()
            return sv  + ov - self.intersection_with(other)            
        else:
            pass
    
    def iou_with(self, other):
        inter = self.intersection_with(other)
        union = self.union_with(other)
        return float(inte) / union 