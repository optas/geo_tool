import sys
import tensorflow as tf
import numpy as np

git_path = '/Users/t_achlp/Documents/Git_Repos/'
sys.path.insert(0, git_path)

from autotensor import autograph
import nn_saliency.src.nn_io as nn_io


IMG_SIZE    = (224, 224)
PART_VIEWS  = 80
NUM_CLASSES = 20

image_pl          = tf.placeholder(tf.float32, shape=(None, IMG_SIZE[0], IMG_SIZE[1], 1))
keep_prob         = tf.placeholder(tf.float32, name='keep_prob')
g                 = autograph()

def compute_inferenceA(input) :
    
    layer = g.conv2d(input, filters=96, field_size=7, stride=2, padding='SAME', name="conv1")\
                        .relu()\
                        .maxpool(kernel=(3,3), stride=(2,2))\
                        .lrn(radius=5, bias=2, alpha=0.0001/5.0, beta=0.75)
            
    layer = g.conv2d(layer, filters=256, field_size=5, stride=2, padding='SAME', name="conv2")\
                        .relu()\
                        .maxpool(kernel=(3,3), stride=(2,2))\
                        .lrn(radius=5, bias=2, alpha=0.0001/5.0, beta=0.75)
                    
    layer = g.conv2d(layer, filters=256, field_size=3, stride=1, padding='SAME', name="conv3")\
                        .relu()

    layer = g.conv2d(layer, filters=256, field_size=3, stride=1, padding='SAME', name="conv4")\
                        .relu()

    layer = g.conv2d(layer, filters=256, field_size=3, stride=1, padding='SAME', name="conv5")\
                        .relu()\
                        .maxpool(kernel=(3,3), stride=(2,2))

    return layer
    
    
def compute_inferenceB(input) :
    
    layer = g.fully_connected(input, 4096, name="fc6")\
                    .relu()\
                    .dropout(keep_prob)
                
    layer = g.fully_connected(layer, 4096, name="fc7")\
                    .relu()\
                    .dropout(keep_prob)
                
    layer = g.fully_connected(layer, NUM_CLASSES, name="fc8")\
                    .relu()\
                    .dropout(keep_prob)
                
    return layer


def compute_3d_inference(input) :
    
        # Reshape to just treat as an array of images.
    in_bundle = tf.reshape(input, [-1, IMG_SIZE[1], IMG_SIZE[0], 1])

        # Compute first stage of inference (treating each image independently)
    inf = compute_inferenceA(g.wrap(in_bundle)).unwrap()

        # Get the shape of the current inference tensor (from the Conv layers)
    inf_shape = inf.get_shape().as_list()
    
        # Reshape the results again to be in buckets of images per solid
    inf = tf.reshape(inf, [-1, PART_VIEWS, inf_shape[1], inf_shape[2], inf_shape[3]])

        # Compute the maximum value across the views and reduce the tensor to that.
    reduce_inf = tf.reduce_max(inf, reduction_indices=1) 
    
        # Compute second stage of inference on the max-reduced data (each across all images for a given shape)
    final_inf = compute_inferenceB(g.wrap(reduce_inf, channels=256)).unwrap()
    
    return final_inf


def mvcnn_gradients():          
    pred_layer        = compute_3d_inference(image_pl)
    max_class         = tf.reduce_max(pred_layer, 1)
    # soft_max          = tf.nn.softmax(pred_layer)
    # num_classes       = pred_layer.get_shape().as_list()[-1]
    # all_other_classes = tf.div(tf.reduce_sum(pred_layer, 1) - max_class, num_classes)
    grads             = tf.gradients(max_class, image_pl)
    return grads 


def initialize_session():
    saver     = tf.train.Saver()
    ckpt_file = '/Users/t_achlp/Documents/DATA/NN/Models/MVCCN/DG/mvcnn_5_20_70_80.ckpt'
    sess = g.session()
    sess.start()
    in_sess = sess.session
    saver.restore(in_sess, ckpt_file)
    return in_sess

def compute_gradients(views_folder, grad_model=None, in_sess=None):
    if grad_model is None:
        grad_model = mvcnn_gradients()
        
    if in_sess is None:
        in_sess =  initialize_session()
                                                                        
    view_tensor, _, _ = nn_io.load_views_of_shape(views_folder, file_format='png', shape_views=PART_VIEWS, reshape_to=IMG_SIZE)
    view_tensor = nn_io.format_image_tensor_for_tf(view_tensor, whiten=True)    
    feed = {image_pl: view_tensor, keep_prob: 1}
    
    return in_sess.run([grad_model[0]], feed_dict=feed), grad_model, in_sess


import os.path as osp
import os
if __name__ == '__main__':    
    top_view_dir = sys.argv[1]
    sub_dirs = [f for f in os.listdir(top_view_dir) if osp.isdir(osp.join(top_view_dir,f))] # expected to be folders of the views.
    sub_dirs = [osp.join(top_view_dir, d) for d in sub_dirs]
    
    res, grad_model, in_sess  = compute_gradients(sub_dirs[0])
    np.savez(osp.join(sub_dirs[0], 'raw_grads'), res[0])
        
    for views in sub_dirs[1:]:
        print views 
        res = compute_gradients(views, grad_model, in_sess)
        np.savez(osp.join(views, 'raw_grads'), res[0])