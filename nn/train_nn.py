import tensorflow as tf
from bvlc_alexnet import AlexNet
import nn
import numpy as np
import os
import cv2

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
    
    test_data_1 = tf.placeholder(tf.float32, shape=(1, 227, 227, 3))
    test_data_2 = tf.placeholder(tf.float32, shape=(1, 227, 227, 3))

    global_step = tf.Variable(0, name = 'global_step', trainable = False)

    net_1 = AlexNet({'data':test_data_1})
    infer_1 = nn.inference( net_1.get_output())
    # net_2 = AlexNet({'data':test_data_2})
    infer_2 = nn.inference( net_1.get_output())
    label = tf.constant(1) 
    loss = nn.triplet_loss(infer_1, infer_2, label)

    training_op = nn.training(loss, 0.5, global_step)

    sess = tf.Session()

    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    net_1.load('bvlc_alexnet.npy', sess)
    # net_2.load('bvlc_alexnet.npy', sess)
    feed_data = {test_data_1: img, test_data_2:img}
    _, loss_v = sess.run([training_op, loss], feed_dict = feed_data)
    # loss_v = sess.run([loss], feed_dict = feed_data)
    print(loss_v)

def main(argv = None):
    train()

if __name__ == '__main__':
    tf.app.run()
