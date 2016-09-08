'''
Created on Jun 14, 2016

@author:    Panos Achlioptas
@contact:   pachlioptas@gmail.com
@copyright: You are free to use, change, or redistribute this code in any
            way you want for non-commercial purposes. 
'''


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_point_cloud(x, y, z, *args, **kwargs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, *args, **kwargs)
    plt.show()