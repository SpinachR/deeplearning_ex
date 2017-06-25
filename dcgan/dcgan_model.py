from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import utils as utils
import tensorflow.contrib.slim as slim


def _batch_norm_params(is_training):
    return {
        "decay": 0.9,
        "epsilon": 1e-5,
        "scale": True,
        "updates_collections": None,
        "is_training": is_training
    }


def _disc_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=utils.lrelu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=_batch_norm_params(is_training),
                        kernel_size=[5, 5], stride=2, padding="SAME",
                        outputs_collections=outputs_collections) as arg_scp:
        return arg_scp


def _gen_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=_batch_norm_params(is_training),
                        outputs_collections=outputs_collections):
        with slim.arg_scope([slim.conv2d_transpose],
                            kernel_size=[3, 3], stride=2, padding="SAME") as arg_scp:

            return arg_scp


def discriminator(inputs, is_training=True, y=None, reuse=None, scope='discriminator'):
    with tf.variable_scope(scope, default_name='discriminator', values=[inputs], reuse=reuse) as scp:
        end_pts_collection = scp.name+'end_pts'
        with slim.arg_scope(_disc_arg_scope(is_training, end_pts_collection)):
            net = slim.conv2d(inputs, 64, normalizer_fn=None, normalizer_params=None, scope='conv0')
            #print('disc:conv0 ', net.get_shape())
            net = slim.conv2d(net, 128, scope='conv1')
            #print('disc:conv1 ', net.get_shape())
            net = slim.conv2d(net, 256, scope="conv2")
            #print('disc:conv2 ', net.get_shape())
            #net = slim.conv2d(net, 512, scope="conv3")
            net = slim.conv2d(net, 1,
                              activation_fn=None,
                              kernel_size=[4, 4], stride=1, padding="VALID",
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="conv4")
            print('disc:conv4 ', net.get_shape())
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            net = tf.squeeze(net, [1, 2], name="squeeze")
            return net, end_pts


def generator(z, is_training=True, y=None, scope='generator'):
    with tf.variable_scope(scope, default_name='generator', values=[z]) as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_gen_arg_scope(is_training, end_pts_collection)):
            net = slim.fully_connected(z, 3 * 3 * 128,
                                       normalizer_fn=None,
                                       normalizer_params=None,
                                       scope="projection")
            net = tf.reshape(net, [-1, 3, 3, 128])
            net = slim.batch_norm(net, scope='batch_norm', **_batch_norm_params(is_training))
            net = slim.conv2d_transpose(net, 64, scope="conv_tp0")
            #print('gen:conv_tp0: ', net.get_shape())
            net = slim.conv2d_transpose(net, 32, scope="conv_tp1")
            #print('gen:conv_tp1: ', net.get_shape())
            #net = slim.conv2d_transpose(net, 64, scope="conv_tp2")
            net = slim.conv2d_transpose(net, 1, padding='VALID',
                                        kernel_size=[6, 6],
                                        activation_fn=tf.nn.tanh,
                                        normalizer_fn=None,
                                        normalizer_params=None,
                                        scope="conv_tp3")
            print('gen:conv_tp3: ', net.get_shape())
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            return net, end_pts

