import tensorflow as tf

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

def inference(data, feature_dim = 1, label_dim = 1):
    leaky_param = tf.constant(0.01, shape = [1], name='leaky_params')

    dim = 1
    for d in data.get_shape().as_list()[1:]:
        dim *= d
    reshape_data = tf.reshape(data, [-1, dim])
    print(reshape_data)
    # data is from pool5
    with tf.variable_scope('fc_layer1') as scope:
        #weights = weight_variable([dim,32])
        weights = variable_with_weight_decay([dim, 100] , 0.1, 'weights', 0.004)
        biases = bias_variable([100])
        h_fc1 = tf.nn.bias_add(tf.matmul(reshape_data, weights),biases)
        hl_relu = add_leaky_relu(h_fc1,leaky_param)
    return hl_relu

def triplet_loss(feature_1, feature_2, labels, radius = 1.0):
    # label is either -1 or 1
    pos_index = max(labels, 0)
    neg_index = max(-labels, 0)
    loss = pos_index * tf.reduce_mean(tf.square(tf.sub(feature_1, feature_2))) + neg_index * tf.maximum(radius*radius, tf.reduce_mean(tf.square(tf.sub(feature_1, feature_2))))
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')

def training(loss, learning_rate, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon = 10.0)
    train_op = optimizer.minimize(loss, global_step = global_step)

    return train_op

