import tensorflow as tf
from bvlc_alexnet_fc7 import AlexNet
import fine_tune_nt
import numpy as np
import os
import time
import cv2
import image_io
import sys

# the dimension of the final layer = feature dim
NN_DIM = 100
LABEL_DIM = 12

TEST_TXT = 'file_list_fine_tune_test_v3.txt'
# TEST_TXT = 'file_list_test.txt'

SHUFFLE_DATA = False
BATCH_SIZE = 1
FEATURE_ROW = 227
FEATURE_COL = 227

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_dir', 'fine_tune_nn_model_logs_v3/','''directory where to save the model''')
tf.app.flags.DEFINE_integer('max_training_iter', 100000,
        '''the max number of training iteration''')

def write_to_file(name_list, value):
    with open("fine_tune_test_res_v3.txt", "w") as f:
        for i in range(len(name_list)):
            f.write(name_list[i])
            f.write(" ")
            f.write(str(value[i]))
            f.write("\n")

def calculate_iter():
    with open(TEST_TXT, 'r') as f:
        s = f.read()
        s_l = s.split('\n')
        total_num = len(s_l)
        if (s_l[total_num - 1] == ""): 
            FLAGS.max_training_iter = int(total_num / BATCH_SIZE) - 1
        else:
            FLAGS.max_training_iter = int(total_num / BATCH_SIZE)
        print(FLAGS.max_training_iter)

def define_graph_config():
    config_proto =  tf.ConfigProto()
    config_proto.gpu_options.per_process_gpu_memory_fraction = 0.5
    return config_proto

def dense_to_one_hot_numpy(labels_dense, num_classes=LABEL_DIM):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def dense_to_one_hot(labels_batch, num_classes = LABEL_DIM):
    sparse_labels = tf.reshape(labels_batch, [-1, 1])
    derived_size = tf.shape(labels_batch)[0]
    indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    outshape = tf.pack([derived_size, num_classes])
    labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
    return labels

def filequeue_to_batch_data(filename_queue, line_reader, batch_size = BATCH_SIZE):
    
    key, next_line = line_reader.read(filename_queue)
    query_image_name, label = tf.decode_csv(
        next_line, [tf.constant([], dtype=tf.string),
            tf.constant([], dtype = tf.int32)], field_delim=" ")
    
    # batch_query_image, batch_label = tf.train.batch(
    #         [query_image_name, label], batch_size=batch_size)

    reverse_channel = True  # for pre-trained purpose
    query_tensor = image_io.read_image(query_image_name, reverse_channel,   
            FEATURE_ROW, FEATURE_COL)

    if SHUFFLE_DATA:
        min_after_dequeue = 100
        capacity = min_after_dequeue + 3 * batch_size
        batch_query_image, batch_label, batch_image_name = tf.train.shuffle_batch(
                [query_tensor, label, query_image_name], batch_size = batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue)
    else:
        batch_query_image, batch_label, batch_image_name = tf.train.batch(
                [query_tensor, label, query_image_name], batch_size=batch_size)
    
    
    return batch_query_image, batch_label, batch_image_name


def train():

    calculate_iter()

    train_filenamequeue=tf.train.string_input_producer([TEST_TXT], shuffle=SHUFFLE_DATA)
    line_reader = tf.TextLineReader()
    batch_image, batch_label, batch_image_name = filequeue_to_batch_data(train_filenamequeue, line_reader)
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    image_data_ph = tf.placeholder(tf.float32, shape = (BATCH_SIZE, FEATURE_ROW, FEATURE_COL, 3))

    net = AlexNet({'data':image_data_ph})

    infer_1 = fine_tune_nt.inference(net.get_output(), LABEL_DIM)

    eva_index = fine_tune_nt.evaluation(infer_1)

    saver = tf.train.Saver()

    config_proto = define_graph_config()
    sess = tf.Session(config = config_proto)

    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    print(ckpt.all_model_checkpoint_paths[-1])
    if ckpt and ckpt.all_model_checkpoint_paths[-1]:
        saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
    else:
        print('no check point, start from begining')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord, sess = sess)
    
    name_list = list()
    dist_list = list()
    for i in xrange(FLAGS.max_training_iter):
        batch_image_v, batch_image_name_v = sess.run([    
            batch_image, batch_image_name])

        feed_data = {image_data_ph: batch_image_v}
        eva_index_v = sess.run(eva_index, feed_dict = feed_data) 

        name_list = name_list + batch_image_name_v.tolist()
        dist_list = dist_list + eva_index_v.tolist()

    write_to_file(name_list, dist_list)

def main(argv = None):
    train()

if __name__ == '__main__':
    tf.app.run()
