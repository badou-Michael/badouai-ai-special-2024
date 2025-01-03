import numpy as np
import tensorflow as tf
import os

class yolo:
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.pre_train = pre_train
        self.anchors = self._get_anchors()
        self.classes = self._get_class()

    #---------------------------------------#
    #   获取种类和先验框
    #---------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    #---------------------------------------#
    #   用于生成层
    #---------------------------------------#
    # 批次归一化
    def _batch_normalization_layer(self, input_layer, training = True, norm_decay = 0.99, norm_epsilon = 1e-3, name = None):
        bn_layer = tf.layers.batch_normalization(
            inputs = input_layer,
            momentum = norm_decay,
            epsilon = norm_epsilon,
            center = True,
            scale = True,
            training = training,
            name = name)
        return tf.nn.leaky_relu(bn_layer, alpha = 0.1)

    # 卷积
    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias = False, strides = 1):
        conv = tf.layers.conv2d(
            inputs = inputs,
            filters = filters_num,
            kernel_size = kernel_size,
            strides = [strides, strides],
            kernel_initializer = tf.glorot_uniform_initializer(),
            padding = ('SAME' if strides == 1 else 'VALID'),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 5e-4),
            use_bias = use_bias,
            name = name)
        return conv

    # 残差卷积
    def _Residual_block(self, inputs, filters_num,
                        blocks_num,
                        conv_index,
                        training = True,
                        norm_decay = 0.99,
                        norm_epsilon = 1e-3,
                        c_name = "conv2d_",
                        n_name = "batch_normalization_"
                        ):
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        layer = self._conv2d_layer(inputs, filters_num, kernel_size = 3, strides = 2, name = c_name + str(conv_index))
        layer = self._batch_normalization_layer(layer, name = n_name + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            layer = self._conv2d_layer(layer, filters_num // 2, kernel_size = 1, strides = 1, name = c_name + str(conv_index))
            layer = self._batch_normalization_layer(layer, name = n_name + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            layer = self._conv2d_layer(layer, filters_num, kernel_size = 3, strides = 1, name = c_name + str(conv_index))
            layer = self._batch_normalization_layer(layer, name = n_name + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index

    #---------------------------------------#
    #   生成_darknet53和逆卷积层
    #---------------------------------------#
    def _darknet53(self, inputs, conv_index,
                   training = True,
                   norm_decay = 0.99,
                   norm_epsilon = 1e-3,
                   c_name = "conv2d_",
                   n_name = "batch_normalization_"
                   ):
        with tf.variable_scope('darknet53'):
            conv = self._conv2d_layer(inputs, filters_num = 32, kernel_size = 3, strides = 1, name = c_name + str(conv_index))
            conv = self._batch_normalization_layer(conv, name = n_name + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 64, blocks_num = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 128, blocks_num = 2, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 256, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            route1 = conv
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 512, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            route2 = conv
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index,  filters_num = 1024, blocks_num = 4, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        return  route1, route2, conv, conv_index

    # 输出两个网络结果
    def _yolo_block(self, inputs, filters_num,
                    out_filters,
                    conv_index,
                    training = True,
                    norm_decay = 0.99,
                    norm_epsilon = 1e-3,
                    c_name = "conv2d_",
                    n_name = "batch_normalization_"
                    ):
        conv = self._conv2d_layer(inputs, filters_num = filters_num, kernel_size = 1, strides = 1, name = c_name + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = n_name + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = c_name + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = n_name + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num, kernel_size = 1, strides = 1, name = c_name + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = n_name + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = c_name + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = n_name + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num, kernel_size = 1, strides = 1, name = c_name + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = n_name + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        route = conv
        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = c_name + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = n_name + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = out_filters, kernel_size = 1, strides = 1, name = c_name + str(conv_index), use_bias = True)
        conv_index += 1
        return route, conv, conv_index

    # 返回三个特征层的内容
    def yolo_inference(self, inputs, num_anchors, num_classes,
                       training = True,
                       c_name = "conv2d_",
                       n_name = "batch_normalization_",
                       r_name = "route_",
                       u_name = "upSample"
                       ):
        conv_index = 1
        conv2d_26, conv2d_43, conv, conv_index = self._darknet53(inputs, conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
        with tf.variable_scope('yolo'):
            #--------------------------------------#
            #   第一个特征层
            #--------------------------------------#
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

            #--------------------------------------#
            #   第二个特征层
            #--------------------------------------#
            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num = 256, kernel_size = 1, strides = 1, name = c_name + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name = n_name + str(conv_index),training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            unSample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]], name= u_name + '0')
            route0 = tf.concat([unSample_0, conv2d_43], axis = -1, name = r_name + '0')
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

            #--------------------------------------#
            #   第三个特征层
            #--------------------------------------#
            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num = 128, kernel_size = 1, strides = 1, name = c_name + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name = n_name + str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            unSample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]], name= u_name + '1')
            route1 = tf.concat([unSample_1, conv2d_26], axis = -1, name = r_name + '1')
            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

        return [conv2d_59, conv2d_67, conv2d_75]

