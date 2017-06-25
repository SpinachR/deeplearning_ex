import os
import time
import numpy as np
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
import dcgan_model as dcgan
import utils
from tensorflow.examples.tutorials.mnist import input_data


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.global_step = tf.Variable(0, name="global_step")
        self._build_model(config)
        self.saver = tf.train.Saver()
        self.mnist = input_data.read_data_sets('MNIST')

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
            tf.summary.scalar("loss_G", self.loss_G)])
        self.summary_writer = tf.summary.FileWriter(config.logdir)

        self.sv = tf.train.Supervisor(
            logdir=config.logdir,
            saver=self.saver,
            summary_op=None,
            summary_writer=self.summary_writer,
            save_model_secs=config.save_model_secs,  # save the model automatically
            checkpoint_basename=config.checkpoint_basename,
            global_step=self.global_step)

        self.sess = self.sv.prepare_or_wait_for_session()

    def _build_model(self, config):
        self.x = tf.placeholder(tf.float32, [None, config.x_dim], name='x')
        self.y = tf.placeholder(tf.float32, [None, config.y_dim], name='y')
        self.z = tf.placeholder(tf.float32, [None, config.z_dim], name='z')
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        print("build_model x: ", self.x.get_shape())

        self.G, self.G_end_pts = dcgan.generator(self.z, self.is_training)

        print("build_model g: ", self.G.get_shape())
        self.D_real, self.D_real_end_pts = dcgan.discriminator(
            tf.reshape(self.x, [-1, config.x_height, config.x_weight, config.x_depth]), self.is_training)
        self.D_fake, self.D_fake_end_pts = dcgan.discriminator(self.G, self.is_training, reuse=True)

        with tf.variable_scope('Loss_D'):
            self.loss_D_real = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=tf.ones_like(self.D_real), logits=self.D_real)
            self.loss_D_fake = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=tf.zeros_like(self.D_fake), logits=self.D_fake)
            self.loss_D = self.loss_D_real + self.loss_D_fake

        with tf.variable_scope("Loss_G"):
            self.loss_G = tf.losses.sigmoid_cross_entropy(
                multi_class_labels=tf.ones_like(self.D_fake), logits=self.D_fake)

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

    def fit(self):
        config = self.config
        for step in range(config.max_steps):
            t1 = time.time()
            z = utils.generate_z(config.batch_size, config.z_dim)
            x = self.mnist.train.next_batch(config.batch_size)

            # train discriminator
            self.sess.run(self.opt_D,
                          feed_dict={self.z: z,
                                     self.x: x[0],
                                     self.is_training: True})

            # train generator
            self.sess.run(self.opt_G,
                          feed_dict={self.z: z,
                                     self.x: x[0],
                                     self.is_training: True})

            self.sess.run(self.opt_G,
                          feed_dict={self.z: z,
                                     self.x: x[0],
                                     self.is_training: True})

            t2 = time.time()
            if (step + 1) % config.summary_every_n_steps == 0:
                summary_feed_dict = {
                    self.z: z, self.is_training: False, self.x: x[0]
                }
                self.make_summary(summary_feed_dict, step + 1)

            if (step + 1) % config.sample_every_n_steps == 0:
                eta = (t2 - t1) * (config.max_steps - step + 1)
                print("Finished {}/{} step, ETA:{:.2f}s"
                      .format(step + 1, config.max_steps, eta))

                _, gen = self.sample(10)
                for i in range(10):
                    imname = os.path.join(config.sampledir,
                                          str(step + 1) + "_" + str(i + 1) + ".jpg")
                    scipy.misc.imsave(imname, gen[i])

    def sample(self, sample_size):
        config = self.config
        z = utils.generate_z(sample_size, config.z_dim)

        return z, self.sample_with_given_z(z)

    def sample_with_given_z(self, z):
        gen = self.sess.run(self.G,
                            feed_dict={self.z: z, self.is_training: False})

        return (gen + 1) / 2.0

    def make_summary(self, feed_dict, step):
        summary = self.sess.run(self.loss_summaries, feed_dict=feed_dict)
        self.sv.summary_computed(self.sess, summary, step)
