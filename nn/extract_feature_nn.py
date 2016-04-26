import tensorflow as tf
from bvlc_alexnet_fc7 import AlexNet
import fine_tune_nt
import numpy as np
import os
import time
import cv2
import image_io

# the dimension of the final layer = feature dim
NN_DIM = 100
LABEL_DIM = 10

# TRAIN_TXT = 'file_list_fine_tune_train.txt'
TRAIN_TXT = 'file_list_fine_tune_test.txt'

TRAIN = True
SHUFFLE_DATA = True
BATCH_SIZE = 1
FEATURE_ROW = 227
FEATURE_COL = 227

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_log_dir','fine_tune_nn_logs',
        '''directory wherer to write event logs''')
tf.app.flags.DEFINE_integer('max_training_iter', 10000,
        '''the max number of training iteration''')
tf.app.flags.DEFINE_float('init_learning_rate',0.001,
        '''initial learning rate''')
tf.app.flags.DEFINE_string('model_dir', 'fine_tune_nn_model_logs','''directory where to save the model''')
tf.app.flags.DEFINE_string('feature_dir', 'f7_dir/','''saved feature dir''')


def define_graph_config():
    config_proto =  tf.ConfigProto()
    config_proto.gpu_options.per_process_gpu_memory_fraction = 0.9
    return config_proto

def calculate_iter():
    with open(TRAIN_TXT, 'r') as f:
        s = f.read()
        s_l = s.split('\n')
        total_num = len(s_l)
        
        FLAGS.max_training_iter = int(total_num / BATCH_SIZE) + 1
        print(FLAGS.max_training_iter)

def write_feature(file_name, feature):
    assert(len(file_name) == len(feature))
    for i in range(len(file_name)):
        f_name = file_name[i].replace(".jpg",".fc7")
        feature.tofile(f_name)

def filequeue_to_batch_data(filename_queue, line_reader, batch_size = BATCH_SIZE):
    
    key, next_line = line_reader.read(filename_queue)
    query_image_name, label = tf.decode_csv(
        next_line, [tf.constant([], dtype=tf.string),
            tf.constant([], dtype = tf.int32)], field_delim=" ")
    
    reverse_channel = True  # for pre-trained purpose
    query_tensor = image_io.read_image(query_image_name, reverse_channel,   
            FEATURE_ROW, FEATURE_COL)

    batch_query_image, batch_label, batch_image_name = tf.train.batch(
            [query_tensor, label, query_image_name], batch_size=batch_size)
    
    
    return batch_query_image, batch_label, batch_image_name


def train():
    calculate_iter()

    train_filenamequeue=tf.train.string_input_producer([TRAIN_TXT], shuffle=SHUFFLE_DATA)

    line_reader = tf.TextLineReader()
    train_batch_image, train_batch_label, batch_image_name = filequeue_to_batch_data(train_filenamequeue, line_reader)

    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    image_data_ph = tf.placeholder(tf.float32, shape = (BATCH_SIZE, FEATURE_ROW, FEATURE_COL, 3))
    label_ph = tf.placeholder(tf.float32, shape = (BATCH_SIZE, LABEL_DIM))

    net = AlexNet({'data':image_data_ph})

    infer = net.get_output()

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    merged_sum = tf.merge_all_summaries()

    config_proto = define_graph_config()
    sess = tf.Session(config = config_proto)

    if TRAIN:
        writer_sum = tf.train.SummaryWriter(FLAGS.train_log_dir,graph_def = sess.graph_def)

    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord, sess = sess)

    if TRAIN:
        for i in xrange(FLAGS.max_training_iter):
            batch_image_v, batch_image_name_v = sess.run([train_batch_image, batch_image_name])
            feed_data = {image_data_ph: batch_image_v}
            infer_v = sess.run(infer, feed_dict = feed_data) 
            write_feature(batch_image_name_v, infer_v)

def main(argv = None):
    # if not os.path.exists(FLAGS.feature_dir):
    #     os.makedirs(FLAGS.feature_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
