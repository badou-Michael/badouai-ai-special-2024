import numpy as np
import tensorflow as tf
import os

class yolo:
    def __init__(self,norm_epsilon,norm_decay,anchors_path,classes_path,pre_train):
        '''introduction:初始化函数
        parameters：
        norm_decay:在预测时计算moving average时的衰减率
        norm_epsilon:方差加上极小的数，防止除以0的情况
        anchors_path: yolo anchor的文件路径
        classes_path:数据集类别对应的文件
        pretrain:是否使用预训练的darknet53模型'''
        self.norm_epsilon=norm_epsilon
        self.norm_decay=norm_decay
        self.anchors_path=anchors_path
        self.classes_path=classes_path
        self.pre_train=pre_train
        self.anchors=self._get_anchors()
        self.classes=self._get_class()

    #获取种类和先验框
    def _get_class(self):
        '''introduction:获取类别名字
        return:coco数据集类别对应的名字'''
        classes_path=os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names=f.readlines()
        class_names=[c.strip() for c in class_names]
        return class_names

    #获取anchors
    def _get_anchors(self):
        anchors_path=os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors=f.readlines()
        anchors=[float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1,2)


    #逻辑如下
    #使用tf.layers.conv2d减少权重和偏置矩阵初始化过程，以及卷积后加上偏置项的操作
    #经过卷积之后需要进行batch norm，最后使用leaky ReLU激活函数
    #根据卷积时的步长，如果卷积的步长为2，则对图像进行降采样
    #比如，输入图片的大小为416*416，卷积核大小为3，若stride为2时，（416 - 3 + 2）/ 2 + 1， 计算结果为208，相当于做了池化层处理
    #因此需要对stride大于1的时候，先进行一个padding操作, 采用四周都padding一维代替'same'方式

    #正则化层
    def _batch_normalization_layer(self,input_layer,name=None,training=True,norm_decay=0.99,norm_epsilon=1e-3):
        '''introduction:对于卷积层提取的feature map进行归一化
        parameters:
        input_layer:输入的四维tensor
        name:batchnorm层的名称
        training：是否为训练过程
        norm_decay:预测时计算moving average的衰减率
        norm_epsilon:方差加上极小的数，防止除以0的情况
        return：
        bn_layer:batch normalization处理后的feature map'''
        bn_layer=tf.layers.batch_normalization(inputs=input_layer,momentum=norm_decay,epsilon=norm_epsilon,
                                               center=True,scale=True,training=training,name=name)
        return tf.nn.leaky_relu(bn_layer,alpha=0.1)

    #卷积层
    def _conv2d_layer(self,inputs,filters_num,kernel_size,name,use_bias=False,strides=1):
        '''introduction:使用tf.layers.conv2d减少权重和偏置矩阵初始化过程，以及卷积后加上偏置项的操作
        parameters：
        inputs:输入变量
        filters_num：卷积核数量
        strides：卷积步长
        name:卷积层名称
        use_bias:是否使用偏置项
        kernel_size:卷积核大小
        return:
        conv:卷积后的feature map
        '''
        conv=tf.layers.conv2d(inputs=inputs,filters=filters_num,kernel_size=kernel_size,
                              strides=[strides,strides],kernel_initializer=tf.glorot_uniform_initializer(),
                              padding=('SAME' if strides==1 else 'VALID'),kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4),
                              use_bias=use_bias,name=name)
        return conv

    #残差卷积,进行一次3X3的卷积，然后保存该卷积layer
    # 再进行一次1X1的卷积和一次3X3的卷积，并把这个结果加上layer作为最后的结果
    def _Residual_block(self,inputs,filters_num,blocks_num,conv_index,training=True,norm_decay=0.99,norm_epsilon=1e-3):
        '''introduction:darknet的残差block，分别采用1*1和3*3卷积核，1*1是为了减少channel的维度
        parameters:
        inputs:输入变量
        filters_num:卷积核数量
        training:是否为训练过程
        blocks_num:block的数量
        conv_index:加载预训练权重，统一命名序号
        weights_dict:加载预训练权重
        return:
        inputs:经过残差训练后的模型
        '''
        #表示在每个维度上的填充大小。第一个维度（batch）不填充。第二个维度（高度）在顶部填充 1 行，底部不填充。
        #第三个维度（宽度）在左侧填充 1 列，右侧不填充。第四个维度（通道）不填充。
        # mode='CONSTANT'：使用常数值（默认是 0）进行填充。
        inputs=tf.pad(inputs,padding=[[0,0],[1,0],[1,0],[0,0]],mode='CONSTANT')
        #应用一个3*3卷积，提取特征并下采样
        layer=self._conv2d_layer(inputs,filters_num,kernel_size=3,strides=2,name='conv2d_'+str(conv_index))
        #对卷积层的输出进行批归一化
        layer=self._batch_normalization_layer(layer,name='batch_normalization'+str(conv_index),training=training,
                                              norm_decay=norm_decay,norm_epsilon=norm_epsilon)
        #增加卷积层的索引，用于下一层的命名
        conv_index+=1

        #循环构建多个残差块。
        for _ in range(blocks_num):
            #保存当前层的输出，作为残差连接的 shortcut。作用：在残差块的最后将 shortcut 加到输出上。
            short_cut=layer
            #应用一个 1x1 卷积层,降低通道数，减少计算量。
            layer=self._conv2d_layer(layer,filters_num//2,kernel_size=1,strides=1,name='conv2d'+str(conv_index))
            layer=self.elf._batch_normalization_layer(layer,name='batch_normalization'+str(conv_index),training=training,
                                              norm_decay=norm_decay,norm_epsilon=norm_epsilon)
            conv_index+=1
            #应用一个 3x3 卷积层,提取特征
            layer = self._conv2d_layer(layer, filters_num, kernel_size=3, strides=1,
                                       name='conv2d' + str(conv_index))
            layer = self.elf._batch_normalization_layer(layer, name='batch_normalization' + str(conv_index),
                                                        training=training,
                                                        norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index+=1
            layer+=short_cut
        return layer,conv_index


    #生成darknet53和逆卷积层
    def _darknet53(self, inputs, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
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
        with tf.variable_scope('darknet53'):
            # 416,416,3 -> 416,416,32
            conv = self._conv2d_layer(inputs, filters_num = 32, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            # 416,416,32 -> 208,208,64
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 64, blocks_num = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            # 208,208,64 -> 104,104,128
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 128, blocks_num = 2, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            # 104,104,128 -> 52,52,256
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 256, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            # route1 = 52,52,256
            route1 = conv
            # 52,52,256 -> 26,26,512
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 512, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            # route2 = 26,26,512
            route2 = conv
            # 26,26,512 -> 13,13,1024
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index,  filters_num = 1024, blocks_num = 4, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            # route3 = 13,13,1024
        return  route1, route2, conv, conv_index

    # 输出两个网络结果
    # 第一个是进行5次卷积后，用于下一次逆卷积的，卷积过程是1X1，3X3，1X1，3X3，1X1
    # 第二个是进行5+2次卷积，作为一个特征层的，卷积过程是1X1，3X3，1X1，3X3，1X1，3X3，1X1
    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        Introduction
        ------------
            yolo3在Darknet53提取的特征层基础上，又加了针对3种不同比例的feature map的block，这样来提高对小物体的检测率
        Parameters
        ----------
            inputs: 输入特征
            filters_num: 卷积核数量
            out_filters: 最后输出层的卷积核数量
            conv_index: 卷积层数序号，方便根据名字加载预训练权重
            training: 是否为训练
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            route: 返回最后一层卷积的前一层结果
            conv: 返回最后一层卷积的结果
            conv_index: conv层计数
        """
        conv = self._conv2d_layer(inputs, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        route = conv
        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = out_filters, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index), use_bias = True)
        conv_index += 1
        return route, conv, conv_index

    # 返回三个特征层的内容
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
            training: 是否为训练模式
        """
        conv_index = 1
        # route1 = 52,52,256、route2 = 26,26,512、route3 = 13,13,1024
        conv2d_26, conv2d_43, conv, conv_index = self._darknet53(inputs, conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
        with tf.variable_scope('yolo'):
            #--------------------------------------#
            #   获得第一个特征层
            #--------------------------------------#
            # conv2d_57 = 13,13,512，conv2d_59 = 13,13,255(3x(80+5))
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

            #--------------------------------------#
            #   获得第二个特征层
            #--------------------------------------#
            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num = 256, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name = "batch_normalization_" + str(conv_index),training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            # unSample_0 = 26,26,256
            unSample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]], name='upSample_0')
            # route0 = 26,26,768
            route0 = tf.concat([unSample_0, conv2d_43], axis = -1, name = 'route_0')
            # conv2d_65 = 52,52,256，conv2d_67 = 26,26,255
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

            #--------------------------------------#
            #   获得第三个特征层
            #--------------------------------------#
            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num = 128, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name = "batch_normalization_" + str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            # unSample_1 = 52,52,128
            unSample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]], name='upSample_1')
            # route1= 52,52,384
            route1 = tf.concat([unSample_1, conv2d_26], axis = -1, name = 'route_1')
            # conv2d_75 = 52,52,255
            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

        return [conv2d_59, conv2d_67, conv2d_75]
