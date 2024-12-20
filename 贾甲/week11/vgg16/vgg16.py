#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author JiaJia time:2024-12-10
import tensorflow as tf

slim = tf.contrib.slim

def vgg_16(inputs,num_classes = 1000,is_training = True,dropout_keep_prob = 0.5,
           spatial_squeeze = True,scrop = 'vgg_16'):
    with tf.varriable_scope(scrop,'vgg_16',[inputs]):
        #01
        net = slim.repeat(inputs,2,slim.conv2d,64,[3,3],scope = 'convl')
        net = slim.max_pool2d(net,[2,2],scope = 'pool1')
        #02
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        #03
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        #04
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        #05
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        #06
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        #07
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')

        #卷积全链接
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        # 由于用卷积的方式模拟全连接层，所以输出需要平铺
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net







