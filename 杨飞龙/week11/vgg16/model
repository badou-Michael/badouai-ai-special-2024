import tensorflow as tf

# 创建slim对象，用于简化网络构建过程
slim = tf.contrib.slim


def vgg_16(inputs,
           num_classes=1000,  # 类别数，默认为1000，适用于ImageNet数据集
           is_training=True,  # 指示是否处于训练模式
           dropout_keep_prob=0.5,  # dropout保留概率
           spatial_squeeze=True,  # 是否进行空间挤压，去除无用维度
           scope='vgg_16'):  # 定义变量作用域
    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        # conv1层：两次3x3卷积，使用ReLU激活，输出64个特征图
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # pool1层：2x2最大池化，步长2
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # conv2层：两次3x3卷积，输出128个特征图
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        # pool2层：2x2最大池化
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # conv3层：三次3x3卷积，输出256个特征图
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        # pool3层：2x2最大池化
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # conv4层：三次3x3卷积，输出512个特征图
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        # pool4层：2x2最大池化
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        # conv5层：三次3x3卷积，输出512个特征图
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        # pool5层：2x2最大池化
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # fc6层：使用7x7卷积核实现全连接层，输出4096个特征
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        # dropout6层：dropout正则化
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')

        # fc7层：使用1x1卷积核实现全连接层，输出4096个特征
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        # dropout7层：dropout正则化
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')

        # fc8层：使用1x1卷积核实现全连接层，输出类别数个特征
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')

        # 如果启用空间挤压，去除无用维度
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net
