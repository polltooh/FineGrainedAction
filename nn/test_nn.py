import tensorflow as tf
from bvlc_alexnet_fc7 import AlexNet
import nt
import numpy as np
import os
import time
import cv2
import image_io
import sys
import math
# the dimension of the final layer = feature dim
NN_DIM = 100

TEST_TXT = 'file_list_test_nba_dunk.txt'
# TEST_TXT = 'file_list_train.txt'
RES_TXT = 'test_res_nba_dunk.txt'
TRAIN = False
SHUFFLE_DATA = False
BATCH_SIZE = 50
FEATURE_ROW = 227
FEATURE_COL = 227
LABEL_DIM = 27
RADIUS = 1.0

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_log_dir','logs',
        '''directory wherer to write event logs''')

# will be changed in the program
tf.app.flags.DEFINE_integer('max_training_iter', 1000,'''the max number of training iteration''')
tf.app.flags.DEFINE_float('init_learning_rate',0.001,'''initial learning rate''')
tf.app.flags.DEFINE_string('model_dir', 'model_logs','''directory where to save the model''')

def write_to_file(name_list, value):
    with open(RES_TXT, "w") as f:
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
        
        FLAGS.max_training_iter = int(total_num / BATCH_SIZE) + 1
        print(FLAGS.max_training_iter)

def filequeue_to_batch_data(filename_queue, line_reader, batch_size = BATCH_SIZE):
    
    key, next_line = line_reader.read(filename_queue)
    query_image_name, retrieve_image_name, label = tf.decode_csv(
        next_line, [tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.string),
            tf.constant([], dtype = tf.int32)], field_delim=" ")
    
    reverse_channel = True  # for pre-trained purpose
    query_tensor = image_io.read_image(query_image_name, reverse_channel,   
            FEATURE_ROW, FEATURE_COL)
    retrieve_tensor = image_io.read_image(retrieve_image_name, reverse_channel,
            FEATURE_ROW, FEATURE_COL)

    if SHUFFLE_DATA:
        min_after_dequeue = 100
        capacity = min_after_dequeue + 3 * batch_size
        batch_query_image, batch_retrieve_image, batch_label, batch_retrieve_image_name = tf.train.shuffle_batch(
                [query_tensor, retrieve_tensor, label, retrieve_image_name], 
                batch_size=batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue)
    else:
        batch_query_image, batch_retrieve_image, batch_label, batch_retrieve_image_name = tf.train.batch(
                [query_tensor, retrieve_tensor, label, retrieve_image_name], 
                batch_size=batch_size)
    
    batch_image = tf.concat(0,[batch_query_image, batch_retrieve_image])

    return batch_image, batch_label, batch_retrieve_image_name


def train():
    calculate_iter()

    train_filenamequeue=tf.train.string_input_producer([TEST_TXT], shuffle=SHUFFLE_DATA)

    line_reader = tf.TextLineReader()
    batch_image, batch_label, batch_image_name = filequeue_to_batch_data(train_filenamequeue, line_reader)

    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    image_data_ph = tf.placeholder(tf.float32, shape = (2 * BATCH_SIZE, FEATURE_ROW, FEATURE_COL, 3))
    label_ph = tf.placeholder(tf.int32, shape = (BATCH_SIZE))

    net = AlexNet({'data':image_data_ph})

    infer = nt.inference2(net.get_output(), NN_DIM)

    eva = nt.evaluation(infer, BATCH_SIZE)

    saver = tf.train.Saver()

    sess = tf.Session()

    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord, sess = sess)

    ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
    print(ckpt.all_model_checkpoint_paths[-1])
    saver.restore(sess, "model_logs/20160417_2125model.ckpt")

    # if ckpt and ckpt.all_model_checkpoint_paths[-1]:
    #     saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
    # else:
    #     print('no check point, start from begining')

    name_list = list()
    dist_list = list()
    for i in xrange(FLAGS.max_training_iter):
        batch_image_v, batch_label_v , batch_image_name_v = sess.run([
                batch_image, batch_label, batch_image_name])

        feed_data = {image_data_ph: batch_image_v, label_ph: batch_label_v}
        eva_v = sess.run(eva, feed_dict = feed_data) 
        name_list = name_list + batch_image_name_v.tolist()
        dist_list = dist_list + eva_v.tolist()
        if i % 100 == 0:
            print("i:%d"%(i))

        # if i != 0 and i % 500 == 0:
        #     curr_time = time.strftime("%Y%m%d_%H%M")
        #     model_name = FLAGS.model_dir + '/' + curr_time + '_iter_' + str(i) + '_model.ckpt'
        #     saver.save(sess,FLAGS.model_dir + '/' + curr_time + 'model.ckpt')
    write_to_file(name_list, dist_list)

def main(argv = None):
    train()

if __name__ == '__main__':
    if (len(sys.argv) >= 2):
        global TEST_TXT
        TEST_TXT = sys.argv[1]
        if (len(sys.argv) > 2):
            global RES_TXT
            RES_TXT = sys.argv[2]

    tf.app.run()
