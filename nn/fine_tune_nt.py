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
    dim = 1
    for d in data.get_shape().as_list()[1:]:
        dim *= d
    print(dim)
    with tf.variable_scope('fc_final') as scope:
        weights = variable_with_weight_decay([dim, nn_dim] , 0.1, 'weights', 0)
        biases = bias_variable([nn_dim])
        h_fc = tf.nn.bias_add(tf.matmul(data, weights), biases)

    return h_fc

def evaluation(infer):
    index = tf.argmax(infer, 1)
    return index

def loss(infer, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(infer,labels,name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', loss)

    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')

def training(loss, learning_rate, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon = 10.0)
    train_op = optimizer.minimize(loss, global_step = global_step)

    return train_op

