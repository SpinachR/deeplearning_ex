import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import input_data


'''## define Initialization Function of Weights and Bias
'''
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name)


'''## define Generator and Discriminator
generator(z): receive input (128) -> output (784)
discrimination(x): receive input (784) -> output (1)
'''
def generator(z):
    h_g1 = tf.nn.relu(tf.add(tf.matmul(z, weights["w_g1"]), biases["b_g1"]))
    h_g2 = tf.nn.sigmoid(tf.add(tf.matmul(h_g1, weights["w_g2"]), biases["b_g2"]))
    return h_g2


def discriminator(x):
    h_d1 = tf.nn.relu(tf.add(tf.matmul(x, weights["w_d1"]), biases["b_d1"]))
    h_d2 = tf.nn.sigmoid(tf.add(tf.matmul(h_d1, weights["w_d2"]), biases["b_d2"]))
    return h_d2



''' ## Sample z (here we use uniform distribution)
'''
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='gray')
    plt.show()


# 0. Read data #################
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 1. Set Paramters   #################
batch_size = 256
g_dim = 128

x_d = tf.placeholder(tf.float32, shape=[None, 784])  # d for Discriminator
x_g = tf.placeholder(tf.float32, shape=[None, 128])  # g for Generator

weights = {
    "w_d1": weight_variable([784, 128], "w_d1"),
    "w_d2": weight_variable([128, 1], "w_d2"),
    "w_g1": weight_variable([128, 256], "w_g1"),
    "w_g2": weight_variable([256, 784], "w_g2")
}

biases = {
    "b_d1": bias_variable([128], "b_d1"),
    "b_d2": bias_variable([1], "b_d2"),
    "b_g1": bias_variable([256], "b_g1"),
    "b_g2": bias_variable([784], "b_g2"),
}

var_d = [weights["w_d1"], weights["w_d2"], biases["b_d1"], biases["b_d2"]]
var_g = [weights["w_g1"], weights["w_g2"], biases["b_g1"], biases["b_g2"]]


# --------------------------------------------------------------------------
# Building the graph
# prepare the real_sample and fake_sample
g_sample = generator(x_g)
d_real = discriminator(x_d)
d_fake = discriminator(g_sample)

# Define the loss (negative sign indicate we need minimization)
d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1. - d_fake))
g_loss = -tf.reduce_mean(tf.log(d_fake))


# --------------------------------------------------------------------------
# Starting Training
print("start training...")

# update discriminator
d_optimizer = tf.train.AdamOptimizer(0.0005).minimize(d_loss, var_list=var_d)
# update generator parameters
g_optimizer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=var_g)


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for step in range(20001):
        batch_x = mnist.train.next_batch(batch_size)[0]
        _, d_loss_train = sess.run([d_optimizer, d_loss], feed_dict={x_d: batch_x, x_g: sample_Z(batch_size, g_dim)})
        _, g_loss_train = sess.run([g_optimizer, g_loss], feed_dict={x_g: sample_Z(batch_size, g_dim)})

        if step <= 1000:
            if step % 100 == 0:
                print("step %d, discriminator loss %.5f" % (step, d_loss_train)),
                print(" | generator loss %.5f" % g_loss_train)
            if step % 1000 == 0:
                g_sample_plot = g_sample.eval(feed_dict={x_g: sample_Z(16, g_dim)})
                plot(g_sample_plot)
        else:
            if step % 1000 == 0:
                print("step %d, discriminator loss %.5f" % (step, d_loss_train)),
                print(" generator loss %.5f" % g_loss_train)
            if step % 2000 == 0:
                g_sample_plot = g_sample.eval(feed_dict={x_g: sample_Z(16, g_dim)})
                plot(g_sample_plot)




