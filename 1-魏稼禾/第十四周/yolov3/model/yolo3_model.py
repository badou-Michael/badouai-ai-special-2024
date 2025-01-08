import tensorflow as tf
import numpy as np
import os

tf.compat.v1.disable_eager_execution()

class yolo:
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path, pre_train):
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.pre_train = pre_train
        self.anchors = self._get_anchors()
        self.classes = self._get_class()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path, encoding="utf8") as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    def _get_anchors(self):
        # 生成注释
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(",")]
        return np.array(anchors).reshape(-1,2)
    
    
    # 层定义
    # batchNorm
    def  _batch_normalization_layer(self, input_layer, name=None, training=True,
                                    norm_decay=0.99, norm_epsilon=1e-3):
        bn_layer = tf.compat.v1.layers.batch_normalization(inputs=input_layer,
            momentum=norm_decay, epsilon=norm_epsilon, center=True, scale=True,
            training=training, name=name)
        return tf.nn.leaky_relu(bn_layer, alpha=0.1)
    
    # conv2d
    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, 
                      use_bias=False, strides=1):
        conv = tf.compat.v1.layers.conv2d(
            inputs=inputs, filters = filters_num,
            kernel_size = kernel_size, strides = [strides, strides], 
            kernel_initializer = tf.compat.v1.glorot_uniform_initializer(),
            padding=("SAME" if strides==1 else "VALID"), 
            kernel_regularizer=tf.keras.regularizers.l1(5e-4), use_bias = use_bias, name=name
        )
        return conv
    
    # residual_block
    # 残差块，每过一层残差块，长宽/2
    # 组成：DBL+res_unit*n
    # DBL 由conv_2d+bn+leaky_relu组成
    def _Residual_block(self, inputs, filters_num, blocks_num, conv_index,
                         training=True, norm_decay=0.99, norm_epsilon=1e-3):
        # 在输入feature map的长宽维度做padding
        inputs = tf.pad(inputs, paddings=[[0,0],[1,0],[1,0],[0,0]], mode="CONSTANT")
        layer = self._conv2d_layer(inputs, filters_num, kernel_size=3, strides=2, name="conv2d_"+str(conv_index))
        layer = self._batch_normalization_layer(layer, name="batch_normalization_"+str(conv_index), 
                                                training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            # 第一层对特征层/2
            layer = self._conv2d_layer(layer, filters_num // 2, kernel_size = 1, strides = 1, name="conv2d_"+str(conv_index))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_"+str(conv_index), 
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            # 第二层恢复原特征层形状
            layer = self._conv2d_layer(layer, filters_num, kernel_size=3, strides=1, name="conv2d_"+str(conv_index))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_"+str(conv_index),
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index
    
    # backbone = darknet53
    # 作用：主干特征提取网络，用来从原始输入图像中提取通用特征
    def _darknet53(self, inputs, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        """
        Introduction
        ------------
            构建yolo3使用的darknet53网络结构
        Parameters
        ----------
            inputs: 模型输入变量
            conv_index: 卷积层数序号，方便根据名字加载预训练权重
            weights_dict: 预训练权重
            training: 是否为训练
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            conv: 经过52层卷积计算之后的结果, 输入图片为416x416x3，则此时输出的结果shape为13x13x1024
            route1: 返回第26层卷积计算结果52x52x256, 供后续使用
            route2: 返回第43层卷积计算结果26x26x512, 供后续使用
            conv_index: 卷积层计数，方便在加载预训练模型时使用
        """
        with tf.compat.v1.variable_scope("darknet53"):
            # 第一层
            # DBL = conv2d+bn+leaky_relu
            # input shape:(bs,416,416,3) -> (bs, 416, 416, 32)
            conv = self._conv2d_layer(inputs, filters_num=32, kernel_size=3, strides=1, name="conv2d_"+str(conv_index))
            conv = self._batch_normalization_layer(conv, name="batch_normalization_"+str(conv_index),training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            # 第二层
            # DBL + res_unit
            # input shape:(bs,416,416,32) -> (bs, 208, 208, 64)
            conv, conv_index = self._Residual_block(conv, filters_num=64, blocks_num=1, conv_index=conv_index, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            # 第三层
            # DBL+res_unit*2
            # input shape:(bs, 208, 208, 64) -> (bs, 104, 104, 128)
            conv, conv_index = self._Residual_block(conv, filters_num=128, blocks_num=2, conv_index=conv_index, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            # 第四层
            # DBL+res_unit*8
            # input shape:(bs, 104, 104, 128) -> (bs, 52, 52, 256)
            conv, conv_index = self._Residual_block(conv, filters_num=256, blocks_num=8, conv_index=conv_index, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route1 = conv   # 返回这一层的输出
            # 第五层
            # DBL+res_unit*8
            # input shape:(bs, 52, 52, 256) -> (bs, 26, 26, 512)
            conv, conv_index = self._Residual_block(conv, filters_num=512, blocks_num=8, conv_index=conv_index, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route2 = conv   # 返回这一层的输出
            # 第六层
            # DBL+res_unit*4
            # input shape:(bs, 26, 26, 512) -> (bs, 13, 13, 1024)
            conv, conv_index = self._Residual_block(conv, filters_num=1024, blocks_num=4, conv_index=conv_index, training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        return route1, route2, conv, conv_index
    
    # yolo_block 包含neck和head中的卷积层
    # neck卷积层 = DBL*5 1x1+3x3+1x1+3x3+1x1
    # neck + head卷积层 = DBL*5 + (DBL+conv) 1x1+3x3+1x1+3x3+1x1+(3x3+1x1-bn)
    # 作用：特征处理与融合网络，用来对backbone提取到的多尺度特征做进一步处理，多尺度融合
    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training=True, norm_decay = 0.99, norm_epsilon = 1e-3):
        # neck conv part
        conv = self._conv2d_layer(inputs, filters_num=filters_num, kernel_size=1, strides=1, name="conv2d_"+str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_"+str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num*2, kernel_size=3, strides=1, name="conv2d_"+str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_"+str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1, name="conv2d_"+str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_"+str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num*2, kernel_size=3, strides=1, name="conv2d_"+str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_"+str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1, name="conv2d_"+str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_"+str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        route = conv
        # head conv part
        # 输出拿来做目标检测的张量
        conv = self._conv2d_layer(conv, filters_num=filters_num*2, kernel_size=3, strides=1, name="conv2d_"+str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_"+str(conv_index), training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=out_filters, kernel_size=1, strides=1, name="conv2d_"+str(conv_index), use_bias=True)
        conv_index += 1
        return route, conv, conv_index
    
    def yolo_inference(self, inputs, num_anchors, num_classes, training=True):
        conv_index = 1
        # route1=52,52,256  route2=26,26,512  conv=13,13,1024
        conv2d_26, conv2d_43, conv, conv_index = self._darknet53(inputs, conv_index, training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
        with tf.compat.v1.variable_scope("yolo"):
            #=====================
            # 第一个特征层
            #=====================
            # 57=52+5, 59=57+2
            # conv2d_57:bs,13,13,512   conv2d_59:bs,13,13,255=(3*85) 
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, num_anchors*(num_classes+5), conv_index, training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            #=====================
            # 第二个特征层
            #=====================
            # (bs,13,13,512)->(bs,13,13,256)
            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num=256, kernel_size=1, strides=1, name="conv2d_"+str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name="batch_normalization_"+str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv_index += 1
            # unSample (bs,13,13,256) -> (bs,26,26,256)
            unSample_0 = tf.compat.v1.image.resize_nearest_neighbor(conv2d_60, [2*tf.shape(conv2d_60)[1], 2*tf.shape(conv2d_60)[1]], name="upSample_0")
            # 256+512=768
            route0 = tf.concat([unSample_0, conv2d_43], axis=-1, name="route_0")
            # conv2d_65:bs,26,26,256  conv2d_67:bs,26,26,255
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors*(num_classes+5),conv_index, training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            #=====================
            # 第三个特征层
            #=====================
            # (bs,26,26,256)->(bs,26,26,128)
            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num=128, kernel_size=1, strides=1, name="conv2d_"+str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name="batch_normalization_"+str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            conv_index += 1
            # (bs,52,52,128)
            unSample_1 = tf.compat.v1.image.resize_nearest_neighbor(conv2d_68, [2*tf.shape(conv2d_68)[1], 2*tf.shape(conv2d_68)[1]], name="upSample_1")
            # 128+256=384
            route1 = tf.concat([unSample_1, conv2d_26], axis=-1, name="route_1")
            _,conv2d_75,_ = self._yolo_block(route1, 128, num_anchors*(num_classes+5), conv_index, training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            
        return [conv2d_59, conv2d_67, conv2d_75]

            