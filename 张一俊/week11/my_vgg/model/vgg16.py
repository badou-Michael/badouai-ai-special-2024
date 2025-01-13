#   vgg16的网络模型构造

import tensorflow as tf

slim = tf.contrib.slim

def vgg_16(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.5, spatial_squeeze=True, scope='vgg_16'):
    """
    :param inputs: # 输入张量，形状为 [batch_size, height, width, channels]
    :param num_classes:  # 输出类别数
    :param is_training: # 是否处于训练模式，用于决定是否使用 dropout
    :param dropout_keep_prob:  # Dropout保留比例
    :param spatial_squeeze: # 是否去除空间维度（高宽）（通常用于最后一层的输出）
    :param scope:  # 网络名称，默认'vgg_16'
    :return net: 输出张量，形状为 [batch_size, num_classes]
    """
    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        # Conv1: 两次3x3卷积，输出通道数64，输入(224,224,3)，输出(224,224,64)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # 2x2池化，输出尺寸减半，为(112,112,64)
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # Conv2：两次3x3，输出通道数128，输入(224,224,64)，输出(112,112,128)
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        # 输出(56,56,128)
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # Conv3：三次3x3，输出通道数256，输出(56,56,256)
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        # 2X2最大池化，输出net为(28,28,256)
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # Conv4：三次3x3卷积，输出通道数512，输出为(28,28,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        # 2X2最大池化，输出net为(14,14,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        # Conv5：三次3x3卷积，输出通道数256，输出为(14,14,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        # 2X2最大池化，输出net为(7,7,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # 卷积模拟全连接层，输出net为(1,1,4096)
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')

        # 卷积模拟全连接层，输出net为(1,1,4096)
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')

        # 输出(1,1,1000)
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')

        # 卷积后的结果不是一维，根据入参铺平
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net