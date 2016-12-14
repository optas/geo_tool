'''
Created on June 14, 2016

@author:    Panayotes Achlioptas
@contact:   pachlioptas @ gmail.com
@copyright: You are free to use, change, or redistribute this code in any
            way you want for non-commercial purposes. 
'''
import os
import re
import cv2
from glob import glob
import matplotlib.animation as animation
import matplotlib.pyplot as plt
try:
    from mayavi import mlab as mayalab
except:
    warnings.warn('Mayavi library was not found. Some graphics utilities will be disabled.')
    

@mlab.show
@mlab.animate(delay=10)
def animate_surface(mesh_surf, output_dir):
    scene  = mesh_surf.scene
    camera = scene.camera
    for i in range(36):
        camera.azimuth(10)
        camera.pitch(5)
        scene.reset_zoom()
        yield        
        scene.save_png(output_dir + '/anim%d.png'%i)

def export_animation(animation_dir, animation_name):
    imagelist = load_animation_images(animation_dir)
    fig = plt.figure()               # make figure
    im  = plt.imshow(imagelist[0])   #TODO fix vmin/vmax for cmap:, cmap=plt.get_cmap('jet'), vmin=np.min(imagelist[0]), vmax=np.max(imagelist[0]))
    # function to update figure
    def updatefig(j):        
        im.set_array(imagelist[j])
        return im,
    # kick off the animation
    ani = animation.FuncAnimation(fig, updatefig, frames=len(imagelist), interval=370, blit=False)
    ani.save(os.path.join(animation_dir, animation_name+'.mp4'))
        
def load_animation_images(folder, file_format='png'):
    searh_pattern = os.path.join(folder, '*.' + file_format)    
    im_names = [n for n in glob(searh_pattern)]
    im_order = list()
    image_data = list()
    for name in im_names:
        m = re.search('(\d+).png$', name)
        im_order.append(int(m.groups()[0]))
    
    visit = ([i[0] for i in sorted(enumerate(im_order), key=lambda x:x[1])])
    
    for im_id in visit:
        image_data.append(cv2.imread(im_names[im_id], cv2.IMREAD_UNCHANGED))                          
    return image_data