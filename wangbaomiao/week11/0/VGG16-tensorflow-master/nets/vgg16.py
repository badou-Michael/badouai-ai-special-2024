# -*- coding: utf-8 -*-
# time: 2024/11/19 18:11
# file: vgg16.py
# author: flame
import tensorflow as tf

slim = tf.contrib.slim

''' 定义 vgg_16 函数，创建 VGG-16 模型。
    参数:
    - inputs: 输入的图像数据
    - num_classes: 输出的类别数，默认为 1000
    - is_training: 是否处于训练模式，默认为 True
    - dropout_keep_prob: Dropout 层保留概率，默认为 0.5
    - spatial_squeeze: 是否对输出进行空间压缩，默认为 True
    - scope: 变量作用域，默认为 'vgg_16'
    返回:
    - net: 最终的输出张量 '''

def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):
    ''' 创建变量作用域，指定作用域名称和输入 '''
    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        ''' 第一层卷积，两次 [3,3] 卷积，输出特征层为 64，输出形状为 (224,224,64) '''
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        ''' 2x2 最大池化，输出形状为 (112,112,64) '''
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        ''' 第二层卷积，两次 [3,3] 卷积，输出特征层为 128，输出形状为 (112,112,128) '''
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        ''' 2x2 最大池化，输出形状为 (56,56,128) '''
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        ''' 第三层卷积，三次 [3,3] 卷积，输出特征层为 256，输出形状为 (56,56,256) '''
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        ''' 2x2 最大池化，输出形状为 (28,28,256) '''
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        ''' 第四层卷积，三次 [3,3] 卷积，输出特征层为 512，输出形状为 (28,28,512) '''
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        ''' 2x2 最大池化，输出形状为 (14,14,512) '''
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        ''' 第五层卷积，三次 [3,3] 卷积，输出特征层为 512，输出形状为 (14,14,512) '''
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        ''' 2x2 最大池化，输出形状为 (7,7,512) '''
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        ''' 利用卷积的方式模拟全连接层，输出形状为 (1,1,4096) '''
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        ''' 添加 Dropout 层，防止过拟合，保留概率为 0.5 '''
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
        ''' 再次利用卷积的方式模拟全连接层，输出形状为 (1,1,4096) '''
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        ''' 添加 Dropout 层，防止过拟合，保留概率为 0.5 '''
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
        ''' 最后一层全连接层，输出形状为 (1,1,num_classes)，激活函数和归一化函数均设为 None '''
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')

        ''' 如果需要空间压缩，对输出进行压缩，去除多余的维度 '''
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        ''' 返回最终的输出张量 '''
        return net
