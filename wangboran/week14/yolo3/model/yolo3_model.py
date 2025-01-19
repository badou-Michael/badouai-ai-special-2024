# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import os

class yolo:
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):
        """
        Introduction
        ------------
            初始化函数
        Parameters
        ----------
            norm_epsilon: 方差加上极小的数,防止除以0的情况
            norm_decay: 在预测时计算moving_average时的衰减率
            anchors_path: yolo anchor 文件路径
            classes_path: 数据集类别对应文件
            pre_train: 是否使用预训练darknet53模型
        """
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.pre_train = pre_train
        self.anchors = self._get_anchors()
        self.classes = self._get_class()

    def _get_class(self):
        """
        Introduction
        ------------
            获取类别名字
        Parameters
        ----------
            class_names: coco数据集类别对应的名字
        """
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        """
        Introduction
        ------------
            获取anchors
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()  # 就一行
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2) # 每两个一组, 进行重组
        return anchors

    def _batch_normalization_layer(self, input_layter, name = None, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        Introduction
        ------------
            对卷积层提取的feature map使用batch normalization
        Parameters
        ----------
            input_layer: 输入的思维tensor
            name: batchnorm层名字
            training: 是否是训练模式
            norm_decay: 在预测时计算moving_average时的衰减率
            norm_epsilon: 方差加上极小的数,防止除以0的情况
        Returns
        -------
            bn_layter: batch normalization处理之后的feature map
        """
        bn_layer = tf.layers.batch_normalization(
            inputs = input_layter, momentum = norm_decay,
            epsilon = norm_epsilon, center = True, scale = True,
            training = training, name = name
        )
        return tf.nn.leaky_relu(bn_layer, alpha = 0.1)

    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias = False, strides = 1):
        """
        Introduction
        ------------
            使用tf.layers.conv2d 减少权重和偏置矩阵初始化过程, 以及卷积后加上偏置项的操作
            经过卷积之后需进行batch norm, 最后使用leaky relu激活函数
            根据卷积时步长大于1时特殊处理, 如果卷积的步长为2, 则对图像进行下采样
                比如输入图片为416x416,卷积核为3,若stride为2
                (416-3+2)/2 + 1 = 208, 相当于做了池化层处理
                因此需要先进行一个padding = 1操作

        Parameters
        ----------
            inputs: 输入变量
            filters_num: 卷积核数量
            kernel_size: 卷积核大小
            name: 卷积层名字
            use_bias: 是否使用偏置项
            strides: 卷积步长
        Returns
        -------
            conv: 卷积之后的feature map
        """
        conv = tf.layers.conv2d(
            inputs = inputs, filters = filters_num, 
            kernel_size = kernel_size, strides = (strides, strides), 
            kernel_initializer = tf.glorot_uniform_initializer(),
            padding = ('same' if strides == 1 else 'valid'),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(5e-4),
            use_bias = use_bias, name = name
        )
        return conv

    def _residual_block(self, inputs, filters_num, blocks_num, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        Introduction
        ------------
            Darknet的残差block,类似resnet的两层卷积结构
            分别采用1x1和3x3卷积核,使用1x1是为了减少channel的维度
        Parameters
        ----------
            inputs: 输入变量
            filters_num: 卷积核数量
            blocks_num: block数量
            conv_index: 卷积层序号, 方便根据名字加载预训练权重
            training: 是否是训练模式
            norm_decay: 在预测时计算moving_average时的衰减率
            norm_epsilon: 方差加上极小的数,防止除以0的情况
        Returns
        -------
            layer: 经过残差网络处理后的结果
            conv_index: 卷积层计数，方便在加载预训练模型时使用
        """
        # 对输入特征图的高度和宽度维度进行填充（在特征图顶部和左侧各填充 1 个像素，其他方向不填充）
        # 第2维(height)：[1, 0]，在高度维度的开头填充 1 行，结尾不填充
        # 第3维 (width)：[1, 0]，在宽度维度的开头填充 1 列，结尾不填充
        # 通过步长（strides=2）下采样时，填充可以避免特征图尺寸减少过多，同时保持有效信息
        inputs = tf.pad(inputs, paddings = [[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        # 对输入特征图进行下采样（减小特征图的宽度和高度）, 特征图的通道数将被转换为 filters_num
        layer = self._conv2d_layer(inputs, filters_num, kernel_size = 3, strides = 2, name = 'conv2d_' + str(conv_index))
        layer = self._batch_normalization_layer(layer, name = 'batch_normalization_' + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            layer = self._conv2d_layer(layer, filters_num // 2, kernel_size = 1, strides = 1, name = 'conv2d_' + str(conv_index))
            layer = self._batch_normalization_layer(layer, name = 'batch_normalization_' + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            layer = self._conv2d_layer(layer, filters_num, kernel_size=3, strides = 1, name = 'conv2d_' + str(conv_index))
            layer = self._batch_normalization_layer(layer, name = 'batch_normalization_' + str(conv_index), training=training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index

    def _darknet53(self, inputs, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        Introduction
        ------------
            构建yolo3使用的darknet53网络结构
        Parameters
        ----------
            inputs: 模型输入变量, 416x416x3 
            conv_index: 卷积层序号, 方便根据名字加载预训练权重
            training: 是否是训练模式
            norm_decay: 在预测时计算moving_average时的衰减率
            norm_epsilon: 方差加上极小的数,防止除以0的情况
        Returns
        -------
            route1: 返回第26层卷积计算结果52x52x256
            route2: 返回第43层卷积计算结果26x26x512
            conv:   返回第52层卷积计算结果13x13x1024
            conv_index: 卷积层计数，方便在加载预训练模型时使用
        """
        with tf.variable_scope('darknet53'):
            # 416,416,3 -> 416,416,32
            conv = self._conv2d_layer(inputs, filters_num = 32, kernel_size = 3, strides = 1, name = 'conv2d_' + str(conv_index))
            conv = self._batch_normalization_layer(conv, name = 'batch_normalization_' + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            # 416,416,32 -> 208.208,64 (2 -> 5)
            conv, conv_index = self._residual_block(conv, conv_index = conv_index, filters_num = 64, blocks_num = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            # 208.208,64 -> 104,104,128 (5 -> 10)
            conv, conv_index = self._residual_block(conv, conv_index = conv_index, filters_num = 128, blocks_num = 2, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            # 104,104,128 -> 52,52,256 (10 -> 27, 注: 当前最后一个层的编号是 conv_index - 1)
            conv, conv_index = self._residual_block(conv, conv_index = conv_index, filters_num = 256, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            # route1 = 52,52,256
            route1 = conv
            # 52,52,256 -> 26,26,512 (27 -> 44)
            conv, conv_index = self._residual_block(conv, conv_index = conv_index, filters_num = 512, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            # route2 = 26,26,512
            route2 = conv
            # 26,26,512 -> 13,13,1024 (44 -> 53)
            conv, conv_index = self._residual_block(conv, conv_index = conv_index, filters_num = 1024, blocks_num = 4, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            # route3 = 13,13,1024
        return route1, route2, conv, conv_index

    # 输出两个网络结果
    # 第一个是进行5次卷积后,用于下一次卷积,卷积过程1x1,3x3,1x1,3x3,1x1
    # 第二个是进行5+2次卷积,作为一个特征层, 卷积过程1x1,3x3,1x1,3x3,1x1,3x1,1x1
    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        Introduction
        ------------
            yolo3在Darknet53提取的特征层基础上, 又增加了针对3种不同比例的feature map的block,以此提高对小物体的检测率
        Parameters
        ----------
            inputs: 输入特征
            filters_num: 卷积核数量
            out_filters: 最后输出层的卷积核数量
            conv_index: 卷积层序号, 方便根据名字加载预训练权重
            training: 是否是训练模式
            norm_decay: 在预测时计算moving_average时的衰减率
            norm_epsilon: 方差加上极小的数,防止除以0的情况
        Returns
        -------
            route: 返回最后一层卷积的前两层结果
            conv: 返回最后一层卷积的结果
            conv_index: conv层计数
        """
        conv = self._conv2d_layer(inputs, filters_num = filters_num, kernel_size = 1, strides = 1, name = 'conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = 'batch_normalization_' + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = 'conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = 'batch_normalization_' + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num, kernel_size = 1, strides = 1, name = 'conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = 'batch_normalization_' + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = 'conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = 'batch_normalization_' + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num, kernel_size = 1, strides = 1, name = 'conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = 'batch_normalization_' + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        route = conv

        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = 'conv2d_' + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = out_filters, kernel_size = 1, strides = 1, name = 'conv2d_' + str(conv_index), use_bias = True)
        conv_index += 1
        return route, conv, conv_index

    def yolo_inference(self, inputs, num_anchors, num_classes, training = True):
        """
        Introduction
        ------------
            构建yolo模型结构
        Parameters
        ----------
            inputs: 模型的输入变量
            num_anchors: 每个grid cell负责检测的anchor数量
            num_classes: 类别数量
            training: 是否是训练模式
        """
        conv_index = 1   # 记录当前是第几个卷积层
        # res1 = 52,52,256, res2 = 26,26,512, res3 = 13,13,1024
        conv2d_26, conv2d_43, conv, conv_index = self._darknet53(inputs, conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
        with tf.variable_scope('yolo'):
            #-------------------------#
            #   获得第一个特征层
            #-------------------------#
            # conv = conv2d_52, conv_index = 53, num_anchors = 3, num_classes = 80
            # conv2d_57 = 13,13,512，conv2d_59 = 13,13,255(3x(80+5))
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            #-------------------------#
            #   获得第二个特征层
            #-------------------------#
            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num=256, kernel_size=1,strides=1, name='conv2d_' + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name='batch_normalization_' + str(conv_index), training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            # upSample0 = 26,26,256
            upSample_0 = tf.image.resize_nearest_neighbor(conv2d_60, 
                        [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[2]],name='upsample_0')
            # route0 = 26,26,768(256+512)
            route0 = tf.concat([upSample_0, conv2d_43], axis = -1, name = 'route_0')
            # conv2d_65 = 26,26,256, conv2d_67 = 26,26,255(3x(80+5))
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

            #-------------------------#
            #   获得第三个特征层
            #-------------------------#
            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num = 128, kernel_size=1, strides = 1, name = 'conv2d_' + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name='batch_normalization_' + str(conv_index), training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            # upSample1 = 52,52,128
            upSample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[2]], name = 'upsample_1')
            # route1 = 52,52,384 (128+256)
            route1 = tf.concat([upSample_1, conv2d_26], axis = -1, name = 'route_1')
            # conv2d_75 = 52,52,255
            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

        return [conv2d_59, conv2d_67, conv2d_75]