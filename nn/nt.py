import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape = shape, 
            stddev = 1e-3, name = 'weights')
    return tf.Variable(initial, name = 'weights')

def bias_variable(shape, value=1e-3):
    initial = tf.constant(value, shape = shape,  name = 'biases')
    return tf.Variable(initial, name = 'biases')

def variable_with_weight_decay(shape, stddev, name, wd):
    var = tf.Variable(tf.truncated_normal(shape = shape, 
        stddev = stddev),
        name = 'weights')
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name = 'weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def add_leaky_relu(hl_tensor, leaky_param):
    return tf.maximum(hl_tensor, tf.mul(leaky_param, hl_tensor))

def inference(data, nn_dim):
    leaky_param = tf.constant(0.01, shape = [1], name='leaky_params')

    dim = 1
    for d in data.get_shape().as_list()[1:]:
        dim *= d
    reshape_data = tf.reshape(data, [-1, dim])
    # data is from pool5

    with tf.variable_scope('fc_layer1') as scope:
        #weights = weight_variable([dim,32])
        weights = variable_with_weight_decay([dim, 1000] , 0.1, 'weights', 0)
        biases = bias_variable([1000])
        h_fc1 = tf.nn.bias_add(tf.matmul(reshape_data, weights), biases)
        h1_relu = add_leaky_relu(h_fc1,leaky_param)

    with tf.variable_scope('fc_layer2') as scope:
        #weights = weight_variable([dim,32])
        weights = variable_with_weight_decay([1000, nn_dim] , 0.1, 'weights', 0)
        biases = bias_variable([nn_dim])
        h_fc2 = tf.nn.bias_add(tf.matmul(h1_relu, weights), biases)
        # h2_relu = add_leaky_relu(h_fc2,leaky_param)

    return h_fc2


def evaluation(infer, batch_size):
    feature_1, feature_2 = tf.split(0,2,infer)
    feature_diff = tf.reduce_sum(tf.square(feature_1 - feature_2), 1)
    return feature_diff

def triplet_loss(infer, labels, batch_size, radius = 1.0):
    feature_1, feature_2 = tf.split(0,2,infer)

    # label is either 0 or 1
    # partition_list = tf.equal(labels,1)
    feature_diff = tf.reduce_sum(tf.square(feature_1 - feature_2), 1)
    feature_list = tf.dynamic_partition(feature_diff, labels, 2)

    # pos_loss = tf.reduce_mean(feature_list[1])
    pos_list = feature_list[1]
    neg_list  = (tf.maximum(0.0, radius * radius - feature_list[0]))
    full_list = tf.concat(0,[pos_list, neg_list])
    loss = tf.reduce_mean(full_list)

    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')

def training(loss, learning_rate, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon = 10.0)

    train_op = optimizer.minimize(loss, global_step = global_step)

    return train_op

