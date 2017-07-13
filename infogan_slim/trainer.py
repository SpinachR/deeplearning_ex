import os
import time
import numpy as np
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import infogan_model as infogan
import utils
from tensorflow.examples.tutorials.mnist import input_data


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.global_step = tf.Variable(0, name="global_step")
        self.mnist = input_data.read_data_sets('MNIST')

        self._build_model(config)

        if not os.path.exists(config.sampledir):
            os.makedirs(config.sampledir)

        if not os.path.exists(config.checkpoint_basename):
            os.makedirs(config.checkpoint_basename)

        if not os.path.exists(config.logdir):
            os.makedirs(config.logdir)

        self.loss_summaries = tf.summary.merge([
            tf.summary.scalar("loss_D_real", self.loss_D_real),
            tf.summary.scalar("loss_D_fake", self.loss_D_fake),
            tf.summary.scalar("loss_D", self.loss_D),
            tf.summary.scalar("loss_G", self.loss_G),
            tf.summary.scalar("loss_Q", self.loss_Q)
        ])

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(config.logdir)
        self.sess = tf.Session()

    def _build_model(self, config):
        self.x = tf.placeholder(tf.float32, [None, config.x_dim], name='x')
        self.z = tf.placeholder(tf.float32, [None, config.z_dim], name='z')
        self.c = tf.placeholder(tf.float32, [None, config.c_dim], name='c')
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.G_sample, self.G_end_pts = infogan.generator(self.z, self.c, self.is_training)
        self.D_real, self.D_real_end_pts = infogan.discriminator(
            tf.reshape(self.x, [-1, config.x_height, config.x_weight, config.x_depth]), self.is_training
        )
        self.D_fake, self.D_fake_end_pts = infogan.discriminator(
            self.G_sample, self.is_training, reuse=True
        )

        self.Q_c_given_x, self.Q_c_end_pts = infogan.encoder_q(self.G_sample, self.is_training)

        with tf.variable_scope('Loss_D'):
            self.loss_D_real = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=tf.ones_like(self.D_real), logits=self.D_real
            )
            self.loss_D_fake = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=tf.zeros_like(self.D_fake), logits=self.D_fake)
            self.loss_D = self.loss_D_real + self.loss_D_fake

        with tf.variable_scope("Loss_G"):
            self.loss_G = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=tf.ones_like(self.D_fake), logits=self.D_fake)

        with tf.variable_scope("Loss_Q"):
            self.loss_Q = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=self.c, logits=self.Q_c_given_x
            )

        with tf.variable_scope("Optimizer_D"):
            vars_D = [var for var in tf.trainable_variables() \
                      if "discriminator" in var.name]
            self.opt_D = tf.train.AdamOptimizer(config.lr,
                                                beta1=config.beta1).minimize(self.loss_D,
                                                                             self.global_step,
                                                                             var_list=vars_D)

        with tf.variable_scope("Optimizer_G"):
            vars_G = [var for var in tf.trainable_variables() \
                      if "generator" in var.name]
            self.opt_G = tf.train.AdamOptimizer(config.lr,
                                                beta1=config.beta1).minimize(self.loss_G, var_list=vars_G)

        with tf.variable_scope("Optimizer_Q"):
            vars_Q = [var for var in tf.trainable_variables() \
                      if "encoder_Q" in var.name]
            self.opt_Q = tf.train.AdamOptimizer(config.lr,
                                                beta1=config.beta1).minimize(self.loss_Q, var_list=vars_Q)

    def fit(self):
        config = self.config
        with self.sess as sess:
            sess.run(tf.global_variables_initializer())
            fig_num = 0
            for step in range(config.max_steps):
                t1 = time.time()
                z = utils.generate_z(config.batch_size, config.z_dim)
                c = utils.generate_c(config.batch_size, config.c_dim)
                x, y = self.mnist.train.next_batch(config.batch_size)


                _, cur_loss_D = sess.run([self.opt_D, self.loss_D], feed_dict={
                    self.z: z, self.x: x, self.c: c, self.is_training: True
                })

                _, cur_loss_G = sess.run([self.opt_G, self.loss_G], feed_dict={
                    self.z: z, self.c: c, self.is_training: True
                })

                _, cur_loss_Q = sess.run([self.opt_Q, self.loss_Q], feed_dict={
                    self.z: z, self.c: c, self.is_training: True
                })

                t2 = time.time()
                if (step+1) % config.summary_every_n_steps == 0:
                    summary_feed_dict = {
                        self.z: z, self.is_training: False, self.x: x, self.c: c
                    }
                    self.make_summary(summary_feed_dict, step + 1)

                if (step + 1) % config.sample_every_n_steps == 0:
                    eta = (t2 - t1) * (config.max_steps - step + 1)
                    print("Finished {}/{} step, ETA:{:.2f}s"
                          .format(step + 1, config.max_steps, eta))

                    z_ = utils.generate_z(16, config.z_dim)
                    c_ = utils.generate_c(16, config.c_dim)
                    samples = sess.run(self.G_sample, feed_dict={self.z: z_, self.c: c_, self.is_training: False})
                    fig = utils.plot(samples)
                    plt.savefig(config.sampledir + '/{}.png'.format(str(fig_num).zfill(3)), bbox_inches='tight')
                    fig_num += 1
                    plt.close(fig)

                if (step + 1) % config.savemodel_every_n_steps == 0:
                    self.saver.save(sess, config.checkpoint_basename, global_step=step+1)

    def make_summary(self, feed_dict, step):
        summary = self.sess.run(self.loss_summaries, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, global_step=step)

