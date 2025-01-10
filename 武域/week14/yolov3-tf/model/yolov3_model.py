import numpy as np
import tensorflow as tf
import os

class yolov3:
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):
        """
        :param norm_epsilon: standard deviation plus small constant to prevent division by zero
        :param norm_decay: decay rate for calculating moving averages
        :param anchors_path: file path of anchors
        :param classes_path: file path of classes (VOC in this case)
        :param pre_train: whether to load pre-trained weights(darknet53)
        """
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.pre_trained = pre_train
        self.anchors = self.get_anchors()
        self.classes = self.get_classes()

    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path, 'r') as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def get_classes(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path, 'r') as f:
            class_names = f.readline()
        class_names = [c.strip() for c in class_names]
        return class_names

    def batch_normalization(self, input_layer, name = None, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        apply batch normalization to input layer, using leaky ReLU to prevent dying ReLU
        :param input_layer: 4 dimension tensor
        :param name: name of this bn layer
        :param training: is training or not
        :param norm_decay: decay rate for calculating moving averages
        :param norm_epsilon: standard deviation plus small constant to prevent division by zero
        :return: feature map after batch normalization
        """
        bn_layer = tf.layers.batch_normalization(inputs = input_layer, momentum = norm_decay, epsilon = norm_epsilon,
                                                 center = True, scale = True, training = training, name = name)
        return tf.nn.leaky_relu(bn_layer, alpha = 0.1)

    def conv2d_layer(self, input_layer, filter_num, kernel_size, name, use_bias = False, stride = 1):
        """
        apply convolutional layer and add bias after convolutional layer.
        apply bn and relu after convolutional layer. Down sampling if stride > 1.
        :param input_layer: input tensor
        :param filter_num: number of convolutional filters
        :param kernel_size: size of convolutional kernel
        :param name: name of this layer
        :param use_bias: whether to use bias or not
        :param stride: stride of convolutional layer
        :return: conv: feature map after convolutional layer
        """
        conv = tf.layers.conv2d(inputs = input_layer, filters = filter_num, kernel_size = kernel_size, strides = [stride, stride],
                                kernel_initializer = tf.glorot_uniform_initializer(),
                                padding = ('SAME' if stride == 1 else 'VALID'), kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 5e-4),
                                use_bias = use_bias, name = name)
        return conv

    def res_block(self, input_layer, filter_num, block_num, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        residual block for darknet53, similar to ResNet, using 1x1(for channel reduction) and 3x3 filters
        :param input_layer: input tensor
        :param filter_num: number of convolutional filters
        :param block_num: number of residual blocks
        :param conv_index: index of convolutional layer for easy loading pre-trained weights
        :param training: is training or not
        :param norm_decay: decay rate for calculating moving averages
        :param norm_epsilon: standard deviation plus small constant to prevent division by zero
        :return: inputs: result after residual block
        """
        inputs = tf.pad(input_layer, [[0, 0], [1, 0], [1, 0], [0, 0]], mode = 'CONSTANT')
        layer = self.conv2d_layer(input_layer = inputs, filter_num = filter_num, kernel_size = 3, stride = 2, name = 'conv2d_' + str(conv_index))
        layer = self.batch_normalization(input_layer = layer, name = 'batch_norm_' + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1

        for _ in range(block_num):
            shortcut = layer
            layer = self.conv2d_layer(layer, filter_num = filter_num // 2, kernel_size = 1, stride = 1, name = 'conv2d_' + str(conv_index))
            layer = self.batch_normalization(layer, name = "batch_norm_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            layer = self.conv2d_layer(layer, filter_num = filter_num, kernel_size = 3, stride = 1, name = 'conv2d_' + str(conv_index))
            layer = self.batch_normalization(layer, name = 'batch_norm_' + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            layer += shortcut

            return layer, conv_index

    def darknet53(self, input_layer, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        construct darknet53 model
        :param input_layer: input tensor
        :param conv_index: index of convolutional layer for easy loading pre-trained weights
        :param training: is training or not
        :param norm_decay: decay rate for calculating moving averages
        :param norm_epsilon: standard deviation plus small constant to prevent division by zero
        :return: conv: result after 52 layers; route1: conv result for 26th layer; route2: conv result for 43 layers; conv_index: index of convolutional layer
        """
        with tf.variable_scope('darknet53'):
            # 426, 426, 3 -> 416, 416, 32
            conv = self.conv2d_layer(input_layer = input_layer, filter_num = 32, kernel_size = 3, stride = 1, name = 'conv2d_' + str(conv_index))
            conv = self.batch_normalization(conv, name = 'batch_norm_' + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            # 416, 416, 32 -> 208, 208, 64
            conv, conv_index = self.res_block(conv, conv_index = conv_index, filter_num = 64, block_num = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            # 208, 208, 64 -> 104, 104, 128
            conv, conv_index = self.res_block(conv, conv_index = conv_index, filter_num = 128, block_num = 2, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            # 104, 104, 128 -> 52, 52, 256
            conv, conv_index = self.res_block(conv, conv_index = conv_index, filter_num = 256, block_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            route1 = conv
            # 52, 52, 256 -> 26, 26, 512
            conv, conv_index = self.res_block(conv, conv_index = conv_index, filter_num = 512, block_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            route2 = conv
            # 26, 26, 512 -> 13, 13, 1024
            conv, conv_index = self.res_block(conv, conv_index = conv_index, filter_num = 1024, block_num = 4, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)

            return route1, route2, conv, conv_index

    def yolo_block(self, inputs, filter_num, out_filters, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        add 3 blocks for feature map to better detect small objects
        :param inputs: input tensor
        :param filter_num: number of convolutional filters
        :param out_filters: number of output filters
        :param conv_index: index of convolutional layer for easy loading pre-trained weights
        :param training: training or not
        :param norm_decay: decay rate for calculating moving averages
        :param norm_epsilon: standard deviation plus small constant to prevent division by zero
        :return: route: conv result before output; conv: conv result; conv_index: index of convolutional layer
        """
        conv = self.conv2d_layer(inputs, filter_num = filter_num, kernel_size = 1, stride = 1, name = 'conv2d_' + str(conv_index))
        conv = self.batch_normalization(conv, name = 'batch_normalization_' + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filter_num = filter_num * 2, kernel_size = 3, stride = 1, name = 'conv2d_' + str(conv_index))
        conv = self.batch_normalization(conv, name = 'batch_normalization_' + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(inputs, filter_num = filter_num, kernel_size = 1, stride = 1, name = 'conv2d_' + str(conv_index))
        conv = self.batch_normalization(conv, name = 'batch_normalization_' + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filter_num = filter_num * 2, kernel_size=3, stride=1, name='conv2d_' + str(conv_index))
        conv = self.batch_normalization(conv, name = 'batch_normalization_' + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(inputs, filter_num=filter_num, kernel_size=1, stride=1,
                                 name='conv2d_' + str(conv_index))
        conv = self.batch_normalization(conv, name='batch_normalization_' + str(conv_index), training=training,
                                        norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        route = conv
        conv = self.conv2d_layer(conv, filter_num=filter_num * 2, kernel_size=3, stride=1,
                                 name='conv2d_' + str(conv_index))
        conv = self.batch_normalization(conv, name='batch_normalization_' + str(conv_index), training=training,
                                        norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filter_num=out_filters, kernel_size=1, stride=1, name='conv2d_' + str(conv_index), use_bias=True)
        conv_index += 1
        return route, conv, conv_index

    def yolo_inference(self, inputs, num_anchors, num_classes, training = True):
        """
        construct yolo model
        :param inputs: input tensor
        :param num_anchors: number of anchors for each grid
        :param num_classes: number of output classes
        :param training: is training or not
        """
        conv_index = 1
        # route 1 = 52, 52, 256; route 2 = 26, 26, 512; route 3 = 13, 13, 1024
        conv2d_26, conv2d_43, conv, conv_index = self.darknet53(inputs, conv_index, training,
                                                                self.norm_decay, self.norm_epsilon)
        with tf.variable_scope('yolo'):
            # -------------------------------
            # get first attribute layer
            # -------------------------------
            # conv2d_57 = 13, 13, 512; conv2d_59 = 13, 13, 255(3*(80+5))(using coco dataset)
            conv2d_57, conv2d_59, conv_index = self.yolo_block(conv, 512, num_anchors * (num_classes + 5), conv_index,
                                                         training, self.norm_decay, self.norm_epsilon)

            #-------------------------------
            # get second attribute layer
            #-------------------------------
            conv2d_60 = self.conv2d_layer(conv2d_57, filter_num=256, kernel_size=1, stride=1, name = 'conv2d_' + str(conv_index))
            conv2d_60 = self.batch_normalization(conv2d_60, name = 'batch_normalization_' + str(conv_index),
                                                 training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv_index += 1
            # upSample_0 = 26, 26, 256
            upSample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2 * tf.shape(conv2d_60)[1],
                                                                      2 * tf.shape(conv2d_60)[1]], name='upSample_0')
            # route0 = 26, 26, 256
            route_0 = tf.concat([upSample_0, conv2d_43], axis=-1, name='route_0')
            # conv2d_65 = 52, 52, 256; conv2d_67 = 26, 26, 255
            conv2d_65, conv2d_67, conv_index = self.yolo_block(route_0, filter_num=256,
                                                               out_filters=num_anchors * (num_classes + 5), conv_index = conv_index, training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)

            # -------------------------------
            # get third attribute layer
            # -------------------------------
            conv2d_68 = self.conv2d_layer(conv2d_65, filter_num=128, kernel_size=1, stride=1, name='conv2d_' + str(conv_index))
            conv2d_68 = self.batch_normalization(conv2d_68, name = 'batch_normalization_' + str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv_index += 1
            # upSample_1 = 52, 52, 128
            upSample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]], name='upSample_1')
            # route1 = 52, 52, 384
            route_1 = tf.concat([upSample_1, conv2d_26], axis=-1, name='route_1')
            # conv2d_75 = 52, 52, 255
            _, conv2d_75, _ = self.yolo_block(route_1, filter_num=128,
                                              out_filters=num_anchors * (num_classes + 5), conv_index=conv_index, training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)

            return [conv2d_59, conv2d_67, conv2d_75]