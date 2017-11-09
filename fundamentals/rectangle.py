'''
Created on December 13, 2016

@author: Panos Achlioptas and Lin Shao
@contact: pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any way you want for non-commercial purposes.
'''

from .. utils import linalg_utils as utils
l2_norm = utils.l2_norm


class Rectangle(object):
    '''
    A class representing a 2D rectangle.
    '''

    def __init__(self, corners):
        '''
        Constructor.
        corners is a numpy array containing 4 non-negative integers
        describing the [xmin, ymin, xmax, ymax] coordinates of the corners of the
        rectangle.
        '''
        self.corners = corners

    def get_corners(self):
        ''' Syntactic sugar to get the corners property into separate variables.
        '''
        c = self.corners
        return c[0], c[1], c[2], c[3]

    def area(self):
        c = self.corners
        return (c[2] - c[0]) * (c[3] - c[1])

    def intersection_with(self, other):
        [sxmin, symin, sxmax, symax] = self.get_corners()
        [oxmin, oymin, oxmax, oymax] = other.get_corners()
        dx = min(sxmax, oxmax) - max(sxmin, oxmin)
        dy = min(symax, oymax) - max(symin, oymin)
        inter = 0
        if (dx > 0) and (dy > 0):
            inter = dx * dy
        return inter

    def diagonal_length(self):
        ''' Returns the length of the diagonal of a rectangle.
        '''
        [xmin, ymin, xmax, ymax] = self.get_corners()
        return l2_norm([xmin - xmax, ymin - ymax])

    def union_with(self, other):
            return self.area() + other.area() - self.intersection_with(other)

    def iou_with(self, other):
        inter = self.intersection_with(other)
        union = self.union_with(other)
        return float(inter) / union

    def overlap_ratio_with(self, other, ratio_type='union'):
        '''
        Returns the overlap ratio between two rectangles. That is the ratio of their area intersection
        and their overlap. If the ratio_type is 'union' then the overlap is the area of their union. If it is min, it
        the min area between them.
        '''
        inter = self.intersection_with(other)
        if ratio_type == 'union':
            union = self.union_with(other)
            return float(inter) / union
        elif ratio_type == 'min':
                return float(inter) / min(self.area(), other.area())
        else:
            ValueError('ratio_type must be either \'union\', or \'min\'.')
