import tensorflow as tf
from bvlc_alexnet import AlexNet
import nn
import numpy as np
import os
import cv2

# the dimension of the final layer = feature dim
NN_DIM = 100

def read_image(path = 'download.jpg'):
    batch_size  = 500
    scale_size  = 258
    crop_size   = 227
    isotropic   = True

    img = cv2.imread(path)
    h, w, c = np.shape(img)
    if isotropic:
        aspect = float(w)/h
        if w<h:
            resize_to = (scale_size, int((1.0/aspect)*scale_size))
        else:
            resize_to = (int(aspect*scale_size), scale_size)
    else:
        resize_to = (scale_size, scale_size)        

    img = cv2.resize(img, resize_to)
    img = img.astype(np.float32)
    mean = np.array([104., 117., 124.])
    img -= mean
    h, w, c = img.shape
    ho, wo = ((h-crop_size)/2, (w-crop_size)/2)
    img = img[ho:ho+crop_size, wo:wo+crop_size, :]
    img = img[None, ...]
    return img

def train():
    img = read_image()
    
    image_data_ph = tf.placeholder(tf.float32, shape=(1, 227, 227, 3))

    infer_ph_1 = tf.placeholder(tf.float32, shape=(1, NN_DIM))
    infer_ph_2 = tf.placeholder(tf.float32, shape=(1, NN_DIM))

    global_step = tf.Variable(0, name = 'global_step', trainable = False)

    net = AlexNet({'data':image_data_ph})
    infer = nn.inference(net.get_output(), NN_DIM)

    label = 1
    loss = nn.triplet_loss(infer_ph_1, infer_ph_2, label)
    # training_op = nn.training(loss, 0.5, global_step)

    sess = tf.Session()

    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    net.load('bvlc_alexnet.npy', sess)

    feed_data = {image_data_ph: img}
    infer_1_v = sess.run(infer, feed_dict = feed_data)
    feed_data = {image_data_ph: img}
    infer_2_v = sess.run(infer, feed_dict = feed_data)
    feed_data = {infer_ph_1: infer_1_v, infer_ph_2: infer_2_v}
    loss_v = sess.run([loss], feed_dict = feed_data) 
    print(loss_v)

def main(argv = None):
    train()

if __name__ == '__main__':
    tf.app.run()
