'''
https://gist.github.com/awjuliani/c9ecd8b37d33d6855cd4ed9aa16ce89f#file-infogan-tutorial-ipynb
'''
import matplotlib
import os
import sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import utils as utils


def _batch_norm_params(is_training):
    return {
        "decay": 0.9,
        "epsilon": 1e-5,
        "scale": True,
        "updates_collections": None,
        "is_training": is_training
    }


def _gen_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=utils.lrelu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=_batch_norm_params(is_training),
                        outputs_collections=outputs_collections):
        with slim.arg_scope([slim.conv2d_transpose],
                            kernel_size=[4, 4], stride=2, padding="SAME") as arg_scp:

            return arg_scp


def _disc_arg_scope(is_training=True, outputs_collections=None):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        activation_fn=utils.lrelu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=_batch_norm_params(is_training),
                        outputs_collections=outputs_collections):
        with slim.arg_scope([slim.conv2d],
                            kernel_size=[4, 4], stride=2, padding="SAME")as arg_scp:
            return arg_scp


def generator(z, c, y=None, is_training=True, scope='generator'):
    '''
    :arg z+c 74
        z: noise. 62-D
        c: latent variables. 1 ten-d categorical code and 2 continuous latent codes
    '''
    with tf.variable_scope(scope, default_name='generator') as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_gen_arg_scope(is_training, end_pts_collection)):
            inputs = tf.concat(values=[z, c], axis=1)   # column concatenate
            net = slim.fully_connected(inputs, 1024, scope='g_fc0')
            net = slim.fully_connected(net, 7*7*128, scope='g_fc1')
            net = tf.reshape(net, shape=[-1, 7, 7, 128])
            net = slim.conv2d_transpose(net, 64, scope="g_conv_tp0")
            net = slim.conv2d_transpose(net, 1,
                                        activation_fn=None,
                                        normalizer_fn=None,
                                        normalizer_params=None,
                                        scope="g_conv_tp1")
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            return net, end_pts


def discriminator(x, is_training=True, reuse=None, scope='discriminator'):
    with tf.variable_scope(scope, default_name='discriminator', reuse=reuse) as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_disc_arg_scope(is_training, end_pts_collection)):
            net = slim.conv2d(x, 64, normalizer_fn=None, normalizer_params=None, scope='d_conv0')
            net = slim.conv2d(net, 128, scope='d_conv1')
            net = tf.reshape(net, [-1, 7 * 7 * 128])
            net = slim.fully_connected(net, 1024, scope='d_fc2')
            net = slim.fully_connected(net, 1,
                              activation_fn=None,
                              normalizer_fn=None,
                              normalizer_params=None,
                              scope="d_fc3")
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            return net, end_pts


def encoder_q(x, is_training=True, scope='encoder_Q'):
    with tf.variable_scope(scope, default_name='encoder_Q') as scp:
        end_pts_collection = scp.name + 'end_pts'
        with slim.arg_scope(_disc_arg_scope(is_training, end_pts_collection)):
            net = slim.conv2d(x, 64, normalizer_fn=None, normalizer_params=None, scope='d_conv0')
            net = slim.conv2d(net, 128, scope='d_conv1')
            net = tf.reshape(net, [-1, 7*7*128])
            net = slim.fully_connected(net, 1024, scope='d_fc2')
            net = slim.fully_connected(net, 12, scope="d_fc3")
            end_pts = slim.utils.convert_collection_to_dict(end_pts_collection)
            return net, end_pts
