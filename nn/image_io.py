import tensorflow as tf
import numpy as np
import cv2


# the image value will be scale to [0,1]
# reverse_channel RGB to BGR 
# row and col have to be pre-defined
# minute the mean of imagenet
def read_image(image_name, reverse_channel, feature_row, feature_col):
    image_bytes = tf.read_file(image_name)
    image_tensor = tf.image.decode_jpeg(image_bytes, channels = 3)
    image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
    image_tensor = tf.image.resize_images(image_tensor, feature_row, feature_col)
    if (reverse_channel):
        dim = tf.constant([False, False, True], dtype = tf.bool)
        image_tensor = tf.reverse(image_tensor, dim)
        image_tensor = image_tensor - [104.0/255, 117.0/255, 124.0/255]
    return (image_tensor) * 255


if __name__ == "__main__":
    image_name = tf.constant("download.jpg", tf.string)
    image = read_image(image_name, 1, 227, 227)
    sess = tf.Session()
    image_v = sess.run(image)
    print(image_v)
    cv2.imshow("test", image_v)
    cv2.waitKey(0)
