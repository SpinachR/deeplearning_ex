# Input data (including pre-processing data)
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Setting hyper-parameter
learning_rate = 0.001
training_iters = 20000
batch_size = 128
display_step = 20


n_input = 784  # input dimensions
n_classes = 10  # num of labels
dropout = 0.8  # probability of dropout


# Define network Input placeholder - X and Y_
X = tf.placeholder(tf.float32, shape=[None, n_input])
Y_ = tf.placeholder(tf.float32, shape=[None, n_classes])
keep_prob = tf.placeholder(tf.float32)


# Define network parameter - wights and bias
weights = {
    'wc1': tf.Variable(tf.truncated_normal([3, 3, 1, 64], stddev=0.1)),
    'wc2': tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1)),
    'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1)),
    'wd1': tf.Variable(tf.truncated_normal([4*4*256, 1024], stddev=0.1)),
    'wd2': tf.Variable(tf.truncated_normal([1024, 1024], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
}
biases = {
    'bc1': tf.Variable(tf.constant(0.1, shape=[64])),
    'bc2': tf.Variable(tf.constant(0.1, shape=[128])),
    'bc3': tf.Variable(tf.constant(0.1, shape=[256])),
    'bd1': tf.Variable(tf.constant(0.1, shape=[1024])),
    'bd2': tf.Variable(tf.constant(0.1, shape=[1024])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}


# Convolution Ops (after activation ops)
def conv2d(name, l_input, w, strides, b):
    z = tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides, padding='SAME'), b)
    return tf.nn.relu(z, name)


# Max Pooling Ops (k for maximize window and strides for shifting steps)
def max_pool(name, l_input, k, strides):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=strides, padding='SAME', name=name)


# Normalization between adjacent layers (local_response_normalization)
# (http://blog.csdn.net/mao_xiao_feng/article/details/53488271)
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


# Alex Net
def alex_net(_X, _weights, _biases, _dropout):
    # reshape to 4-D
    _X = tf.reshape(_X, shape=[-1, 28, 28, 1])
    conv_strides = [1, 1, 1, 1]
    pooling_strides = [1, 2, 2, 1]

    # conv1 -> maxPooling1 -> normalization1 ->dropout1
    conv1 = conv2d('conv1', _X, _weights['wc1'], conv_strides, _biases['bc1'])
    pool1 = max_pool('pool1', conv1, k=2, strides=pooling_strides)
    norm1 = norm('norm1', pool1, lsize=4)
    norm1 = tf.nn.dropout(norm1, _dropout)

    # conv2 -> maxPooling2 -> normalization2 ->dropout2
    conv2 = conv2d('conv2', norm1, _weights['wc2'], conv_strides, _biases['bc2'])
    pool2 = max_pool('pool2', conv2, k=2, strides=pooling_strides)
    norm2 = norm('norm2', pool2, lsize=4)
    norm2 = tf.nn.dropout(norm2, _dropout)

    # conv3 -> maxPooling3 -> normalization3 ->dropout3
    conv3 = conv2d('conv3', norm2, _weights['wc3'], conv_strides, _biases['bc3'])
    pool3 = max_pool('pool3', conv3, k=2, strides=pooling_strides)
    norm3 = norm('norm3', pool3, lsize=4)
    norm3 = tf.nn.dropout(norm3, _dropout)

    # full connect1
    dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
    # full connect2
    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')  # Relu activation

    # output
    out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return out



'''
1. start to build graph
________________________________________
'''

pred = alex_net(X, weights, biases, keep_prob)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y_))
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


# Before training, we'd better build eval network
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


'''
2. start to train
________________________________________
'''
# 初始化所有的共享变量
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        sess.run(train, feed_dict={X: batch_xs, Y_: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # calc train accuracy
            acc = sess.run(accuracy, feed_dict={X: batch_xs, Y_: batch_ys, keep_prob: 1.})
            # calc loss
            loss = sess.run(loss, feed_dict={X: batch_xs, Y_: batch_ys, keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    # calc test accuracy
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images[:256], Y_: mnist.test.labels[:256], keep_prob: 1.}))