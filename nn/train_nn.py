import tensorflow as tf
from bvlc_alexnet import AlexNet
import nt
import numpy as np
import os
import cv2
import image_io

# the dimension of the final layer = feature dim
NN_DIM = 100

TRAIN_TXT = 'file_list_train.txt'
TEST_TXT = 'file_list_test.txt'

TRAIN = True
SHUFFLE_DATA = False
BATCH_SIZE = 50
FEATURE_ROW = 227
FEATURE_COL = 227
LABEL_DIM = 27
RADIUS = 1.0

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_log_dir','/home/mscvadmin/action/FineGrainedAction/nn/logs',
        '''directory wherer to write event logs''')
tf.app.flags.DEFINE_integer('max_training_iter', 10000,
        '''the max number of training iteration''')
tf.app.flags.DEFINE_float('init_learning_rate',0.001,
        '''initial learning rate''')
tf.app.flags.DEFINE_string('model_dir', '/home/mscvadmin/action/FineGrainedAction/nn/model_logs','''directory where to save the model''')



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
    image_data_ph = tf.placeholder(tf.float32, shape = (2 * BATCH_SIZE, FEATURE_ROW, FEATURE_COL, 3))
    label_ph = tf.placeholder(tf.int32, shape = (BATCH_SIZE))

    net = AlexNet({'data':image_data_ph})

    infer = nt.inference(net.get_output(), NN_DIM)

    tloss = nt.triplet_loss(infer, label_ph, BATCH_SIZE, RADIUS)

    lr = FLAGS.init_learning_rate
    train_op = nt.training(tloss, lr, global_step) 

    saver = tf.train.Saver()

    sess = tf.Session()
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

            feed_data = {image_data_ph: batch_image_v, label_ph: batch_label_v}
            loss_v,_ = sess.run([tloss, train_op], feed_dict = feed_data) 
            if i % 100 == 0:
                print("i:%d, loss:%f"%(i,loss_v))

            if i != 0 and i % 1000 == 0:
                curr_time = time.strftime("%Y%m%d_%H%M")
                model_name = FLAGS.model_dir + '/' + curr_time + '_iter_' + str(i) + '_model.ckpt'
                saver.save(sess,FLAGS.model_dir + '/' + curr_time + 'model.ckpt')


def main(argv = None):
    train()

if __name__ == '__main__':
    tf.app.run()
