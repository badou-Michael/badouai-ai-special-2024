import numpy as np
import tensorflow as tf
import os

class yolo:
    def __init__(self, norm_epsilon, norm_decay,anchors_path, classes_path, pre_train):
        #初始化函数
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.classes_path = classes_path
        self.anchors_path = anchors_path
        self.pre_train = pre_train
        self.anchors = self._get_anchors()
        self.classes = self._get_class()


    #获取种类和先验框
    def _get_class(self):
        #获取类别名称
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        #获取anchors
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _batch_normalization_layer(self, inputs_layer, name=None, training=True,
                                   norm_decay = 0.99, norm_epsilon = 1e-3):
        #对卷积层提取的feature map使用batch normalization
        bn_layer = tf.layers.batch_normalization(inputs=inputs_layer,
            momentum = norm_decay, epsilon = norm_epsilon, center=True,
            scale=True, training=training, name = name)
        return tf.nn.leaky_relu(bn_layer, alpha = 0.1)

    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias=False, strides=1):
        """
           使用tf.layers.conv2d减少权重和偏置矩阵初始化过程，以卷积后加上偏置项的操作
           经过卷积之后需要进行batch norm,最后使用leaky ReLU激活函数
           根据卷积时的步长，如果卷积的步长为2，则对图像进行降条样
        """
        conv = tf.layers.conv2d(
            inputs=inputs, filters=filters_num,
            kernel_size=kernel_size, strides=[strides, strides], kernel_initializer=tf.glorot_uniform_initializer(),
            padding=('SAME' if strides == 1 else 'VALID'),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4), use_bias=use_bias, name=name)
        return conv


    #这个用来进行残差卷积的
    def _Residual_block(self, inputs, filters_num, blocks_num, conv_index,
                        training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        #Darknet的残差block,类似resnet的两层卷积结构，分别采用1*1和3*3的卷积核，使用1*1是为了减少channel的维度
        inputs = tf.pad(inputs, paddings=[[0,0],[1,0],[1,0],[0,0]], mode = 'CONSTANT')
        layer = self._conv2d_layer(inputs, filters_num, kernel_size = 3, strides = 2, name = "conv2d_" + str(conv_index))
        layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            layer = self._conv2d_layer(layer, filters_num//2, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            layer = self._conv2d_layer(layer, filters_num, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index

    #生成_darknet53和逆卷积层
    def _darknet53(self, inputs, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        #构建yolo3使用的darknet53网络结构
        with tf.variable_scope('darknet53'):
            #416,416,3 -> 416, 416,32
            conv = self._conv2d_layer(inputs, filters_num = 32, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            #416,416,32 ->208,208,64
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num=64, blocks_num=1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            # 208,208,64 ->104,104,128
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=128, blocks_num=2,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            #104,104,128 -> 52,52,256
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=256, blocks_num=8,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route1 = conv
            #52,52,256->26,26,512
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=512, blocks_num=8,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route2 = conv
            #26,26,512 -> 13,13,1024
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=1024, blocks_num=4,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            #route3 = 13,13,1024
            return route1, route2, conv, conv_index

    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
          yolo3在darknet35提取的特征层基础上，又加了针对3种不同比例的feature map的block,
          这样来提高对小物体的检测率
        """
        conv = self._conv2d_layer(inputs, filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num , kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num*2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        route = conv
        conv = self._conv2d_layer(conv, filters_num=filters_num*2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=out_filters, kernel_size=1, strides=1, name="conv2d_" + str(conv_index), use_bias=True)
        conv_index += 1
        return route, conv, conv_index

    #反回三个特片层的内容
    def yolo_inference(self,inputs, num_anchors, num_classes, training = True):
        """
         构建yolo模型结构
        """
        conv_index = 1
        conv2d_26, conv2d_43, conv, conv_index = self._darknet53(inputs, conv_index,training= training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
        with tf.variable_scope("yolo"):
            #获得第一个特征层
            #conv2d_57 = 13,13,512, conv2d_59 = 13,13,255(3*(80+5))
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, num_anchors * (num_classes + 5), conv_index = conv_index, training= training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

            #获得第二个特征层
            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num=256, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name = "batch_normalization_" + str(conv_index),training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            #unSample_0 = 25,25,256
            unSample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2*tf.shape(conv2d_60)[1], 2*tf.shape(conv2d_60)[1]], name='upSample_0')
            #route0 = 26,26,256
            route0 = tf.concat([unSample_0, conv2d_43], axis=-1, name='route_0')
            #conv2d_65 = 52,52,256, conv2d_67 = 26,26,255
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors * (num_classes + 5), conv_index=conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

            #获得第三个特征层
            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num=128, kernel_size=1, strides=1, name="conv2d_" + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name = "batch_normalization_" + str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1

            unSample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]], name='upSample_1')
            route1 = tf.concat([unSample_1, conv2d_26], axis=-1, name='route_1')

            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors * (num_classes +5),  conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
        return [conv2d_59, conv2d_67, conv2d_75]

