#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：vgg16.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/12/12 17:36
'''

import tensorflow.compat.v1 as tf
import tf_slim as slim

# slim = tf_slim.contrib.slim
tf.disable_v2_behavior()


def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope="vgg_16"):
    with tf.variable_scope(scope, "vgg_16", [inputs]):
        # output shape:(224,224,64)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope="conv1")
        # output shape:(112,112,64)
        net = slim.max_pool2d(net, [2, 2], stride=2, scope="pool1")

        # output shape:(112,112,128)
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope="conv2")
        # output shape:(56,56,128)
        net = slim.max_pool2d(net, [2, 2], stride=2, scope="pool2")

        # output shape:(56,56,256)
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope="conv3")
        # output shape:(28,28,256)
        net = slim.max_pool2d(net, [2, 2], stride=2, scope="pool3")

        # output shape:(28,28,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope="conv4")
        # output shape:(14,14,512)
        net = slim.max_pool2d(net, [2, 2], stride=2, scope="pool4")

        # output shape:(14,14,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope="conv5")
        # output shape:(7,7,512)
        net = slim.max_pool2d(net, [2, 2], stride=2, scope="pool5")

        # output shape:(1,1,4096)
        net = slim.conv2d(net, 4096, [7, 7], padding="VALID", scope="fc6")
        net = slim.dropout(net, dropout_keep_prob,
                           is_training=is_training, scope="dropout6")
        # output shape:(1,1,4096)
        net = slim.conv2d(net, 4096, [1, 1], padding="VALID", scope="fc7")
        net = slim.dropout(net, dropout_keep_prob,
                           is_training=is_training, scope="dropout7")
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope="fc8")

        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name="fc8/squeezed")
        return net