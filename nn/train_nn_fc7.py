import tensorflow as tf
from bvlc_alexnet_fc7 import AlexNet
import nt
import numpy as np
import os
import time
import cv2
import image_io
import utility_function as uf

# the dimension of the final layer = feature dim
NN_DIM = 1000

TRAIN_TXT = 'file_list_train_fc7.txt'
TEST_TXT = 'file_list_test.txt'

TRAIN = True
SHUFFLE_DATA = True
BATCH_SIZE = 50
FEATURE_ROW = 227
FEATURE_COL = 227
RADIUS = 1.0

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_log_dir','logs_fc7', '''directory wherer to write event logs''')
tf.app.flags.DEFINE_integer('max_training_iter', 100000, '''the max number of training iteration''')
tf.app.flags.DEFINE_float('init_learning_rate',0.001, '''initial learning rate''')
tf.app.flags.DEFINE_string('model_dir', 'model_logs_fc7','''directory where to save the model''')



def define_graph_config():
    config_proto =  tf.ConfigProto()
    config_proto.gpu_options.per_process_gpu_memory_fraction = 0.2
    return config_proto

def filequeue_to_batch_data(filename_queue, line_reader, batch_size = BATCH_SIZE):
    
    key, next_line = line_reader.read(filename_queue)
    query_image_name, retrieve_image_name, label = tf.decode_csv(
        next_line, [tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.string),
            tf.constant([], dtype = tf.int32)], field_delim=" ")

    query_tensor = uf.read_binary(query_image_name, 4096)
    retrieve_tensor = uf.read_binary(retrieve_image_name, 4096)

    if SHUFFLE_DATA:
        min_after_dequeue = 100
        capacity = min_after_dequeue + 3 * batch_size
        batch_query_image, batch_retrieve_image, batch_label = tf.train.shuffle_batch(
                [query_tensor, retrieve_tensor, label], batch_size=batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue)
    else:
        batch_query_image, batch_retrieve_image, batch_label = tf.train.batch(
                [query_tensor, retrieve_tensor, label], batch_size=batch_size)
    
    
    batch_image = tf.concat(0,[batch_query_image, batch_retrieve_image])
    return batch_image, batch_label


def train():
    train_filenamequeue=tf.train.string_input_producer([TRAIN_TXT], shuffle=SHUFFLE_DATA)

    line_reader = tf.TextLineReader()
    train_batch_image, train_batch_label = filequeue_to_batch_data(train_filenamequeue, line_reader)

    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    feature_ph = tf.placeholder(tf.float32, shape = (2 * BATCH_SIZE, 4096))
    label_ph = tf.placeholder(tf.int32, shape = (BATCH_SIZE))

    infer = nt.inference3(feature_ph , NN_DIM)

    tloss = nt.triplet_loss(infer, label_ph, BATCH_SIZE, RADIUS)

    tf.scalar_summary('loss', tloss)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    merged_sum =tf.merge_all_summaries()

    lr = FLAGS.init_learning_rate
    train_op = nt.training(tloss, lr, global_step) 

    saver = tf.train.Saver()

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
            batch_image_v, batch_label_v = sess.run([    
                train_batch_image, train_batch_label])

            feed_data = {feature_ph: batch_image_v, label_ph: batch_label_v}
            loss_v,_,merged_sum_v = sess.run([tloss, train_op, merged_sum], feed_dict = feed_data) 
            if i % 100 == 0:
                print("i:%d, loss:%f"%(i,loss_v))
                if i != 0:
                    writer_sum.add_summary(merged_sum_v, i)

            if i != 0 and i % 500 == 0:
                curr_time = time.strftime("%Y%m%d_%H%M")
                model_name = FLAGS.model_dir + '/' + curr_time + '_iter_' + str(i) + '_model.ckpt'
                saver.save(sess,FLAGS.model_dir + '/' + curr_time + 'model.ckpt')


def main(argv = None):
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    if not os.path.exists(FLAGS.train_log_dir):
        os.makedirs(FLAGS.train_log_dir)
    train()

if __name__ == '__main__':
    tf.app.run()
