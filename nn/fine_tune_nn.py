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
LABEL_DIM = 3

TRAIN_TXT = 'file_list_fine_tune_train.txt'
# TEST_TXT = 'file_list_test.txt'

TRAIN = True
SHUFFLE_DATA = True
BATCH_SIZE = 50
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

def define_graph_config():
    config_proto =  tf.ConfigProto()
    config_proto.gpu_options.per_process_gpu_memory_fraction = 0.9
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
        batch_query_image, batch_label = tf.train.shuffle_batch(
                [query_tensor, label], batch_size = batch_size, capacity=capacity,
                min_after_dequeue=min_after_dequeue)
    else:
        batch_query_image, batch_label = tf.train.batch(
                [query_tensor, label], batch_size=batch_size)
    
    
    return batch_query_image, batch_label


def train():
    train_filenamequeue=tf.train.string_input_producer([TRAIN_TXT], shuffle=SHUFFLE_DATA)

    line_reader = tf.TextLineReader()
    train_batch_image, train_batch_label = filequeue_to_batch_data(train_filenamequeue, line_reader)

    train_batch_label_one_hot = dense_to_one_hot(train_batch_label, LABEL_DIM)

    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    image_data_ph = tf.placeholder(tf.float32, shape = (BATCH_SIZE, FEATURE_ROW, FEATURE_COL, 3))
    label_ph = tf.placeholder(tf.float32, shape = (BATCH_SIZE, LABEL_DIM))

    net = AlexNet({'data':image_data_ph})

    infer = fine_tune_nt.inference(net.get_output(), LABEL_DIM)

    loss = fine_tune_nt.loss(infer, label_ph)

    tf.scalar_summary('loss', loss)

    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    merged_sum = tf.merge_all_summaries()

    lr = FLAGS.init_learning_rate
    train_op = fine_tune_nt.training(loss, lr, global_step)

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
                train_batch_image, train_batch_label_one_hot])

            feed_data = {image_data_ph: batch_image_v, label_ph: batch_label_v}
            loss_v,_ = sess.run([loss, train_op], feed_dict = feed_data) 
            if i % 100 == 0:
                print("i:%d, loss:%f"%(i,loss_v))

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
