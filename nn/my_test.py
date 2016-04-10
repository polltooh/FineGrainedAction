import tensorflow as tf
import cv2
import numpy as np
from bvlc_alexnet import AlexNet

def read_image(model, path):
    img = cv2.imread(path)
    h, w, c = np.shape(img)
    scale_size = model.scale_size
    crop_size = model.crop_size
    assert c==3
    if model.isotropic:
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



def test_imagenet():
    test_data = tf.placeholder(tf.float32, shape=(1, 227, 227, 3))
    net = AlexNet({'data':test_data})
    image = read_image(net, "download.jpg")
    with tf.Session() as sesh:
        net.load('bvlc_alexnet.npy', sesh)
        prob_v = sesh.run(net.get_output(), feed_dict={test_data:image})
        print(prob_v.shape)

def test_tensor():

    t = tf.constant([1,1,3,3],tf.int32)
    t1, t2 = tf.split(0, 2, t, name = 'split')
    sess = tf.Session()
    a_v = sess.run(t1)
    print(a_v)


def main():
    test_tensor()
    # test_imagenet()

if __name__ == "__main__":
    main()
