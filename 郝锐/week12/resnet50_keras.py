#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/12/21 11:01
# @Author: Gift
# @File  : resnet50_keras.py 
# @IDE   : PyCharm
#-------------------------------------------------------------#
#   ResNet50的网络部分
# resnet50包括俩个基本模块                                       #
#   1. conv_block：由三个卷积层组成，每个卷积层后面都跟着一个BN层和激活函数#
#   2. identity_block：由三个卷积层组成，每个卷积层后面都跟着一个BN层和激活函数#
#-------------------------------------------------------------#
import numpy as np
from keras import layers
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, ZeroPadding2D, Add, Input, Flatten, \
    Dense, AveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    :param input_tensor:输入的张量，代表输入到该模块的数据。
    :param kernel_size:卷积核的尺寸，用于指定中间卷积层的卷积核大小
    :param filters:一个包含三个元素的元组或列表，分别表示三个卷积层的滤波器（卷积核）数量。
    :param stage:一个整数，通常用于标识当前模块处于整个网络的哪个阶段，用于构建层的命名。
    :param block:一个字符串，用于进一步细分同一阶段内不同的模块，同样用于命名。
    :param strides:一个二元组，默认值为(2, 2)，用于指定第一个卷积层的步长，常用来进行下采样操作。
    :return:
    """
    #将传入的filters参数解包为三个变量filters1、filters2、filters3，分别对应模块中三个不同卷积层的滤波器数量。
    filters1, filters2, filters3 = filters
    #用于卷积层和批归一化层命名，有助于在模型可视化等场景下清晰地识别各个层。
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    #第一个卷积层
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    #批归一化层
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    #激活函数
    x = Activation('relu')(x)
    #第二个卷积层
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    #批归一化层
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    #激活函数
    x = Activation('relu')(x)
    #第三个卷积层
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    #批归一化层
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    #构建一个直接从输入到模块末尾的快捷方式
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
    #特征融合与激活
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """

    :param input_tensor: 输入的张量，代表输入到该模块的数据。
    :param kernel_size: 卷积核的尺寸，用于指定中间卷积层的卷积核大小
    :param filters: 一个包含三个元素的元组或列表，分别表示三个卷积层的滤波器（卷积核）数量。
    :param stage: 一个整数，通常用于标识当前模块处于整个网络的哪个阶段，用于构建层的命名。
    :param block: 一个字符串，用于进一步细分同一阶段内不同的模块，同样用于命名。
    :return:
    """
    #将传入的filters参数解包为三个变量filters1、filters2、filters3，分别对应模块中三个不同卷积层的滤波器数量。
    filters1, filters2, filters3 = filters
    #用于卷积层和批归一化层命名，有助于在模型可视化等场景下清晰地识别各个层。
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    #第一个卷积层
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    #批归一化层
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    #激活函数
    x = Activation('relu')(x)
    #第二个卷积层
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    #批归一化层
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    #激活函数
    x = Activation('relu')(x)
    #第三个卷积层
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    #批归一化层
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    #特征融合与激活-identity block不需要shortcut,直接将输入与输出相加即可
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x
#ResNet50 深度卷积神经网络模型
def ResNet50(input_shape=[224, 224, 3], classes=1000):
    """

    :param input_shape: 输入类型为224*224的3通道图片
    :param classes: 输出分类为1000
    :return:
    """
    #定义整个网络的数据入口，后续的所有操作都将基于这个输入进行处理。
    img_input = Input(shape=input_shape)
    #ZeroPadding2D层，用于在输入数据的边缘填充0，以保持输入数据的尺寸不变。
    x = ZeroPadding2D((3, 3))(img_input)
    #第一个卷积层-64个卷积核-卷积核大小7*7-步长2*2-命名为conv1
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    #批归一化层
    x = BatchNormalization(name='bn_conv1')(x)
    #激活函数
    x = Activation('relu')(x)
    #最大池化层-池化核大小3*3-步长2*2
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    #根据resnet50的架构，进行卷积操作
    #1个conv_block和俩个identity_block
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    #1个conv_block和三个identity_block
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    #1个conv_block和5个identity_block
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    #1个conv_block和2个identity_block
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    #平均池化层-池化核大小7*7-步长1*1-减少数据维度，同时保留整体的特征信息。
    x = AveragePooling2D((7,7), name='avg_pool')(x)
    #数据拍扁成1维
    x = Flatten()(x)
    #全连接层-输出1000个节点-激活函数softmax
    x = Dense(classes, activation='softmax', name='fc1000')(x)
    #定义模型
    model = Model(img_input, x, name='resnet50')
    #加载预训练权重
    model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    return model
if __name__ == '__main__':
    model = ResNet50()
    model.summary()
    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    #图像转数组
    x = image.img_to_array(img)
    #增加一个维度
    x = np.expand_dims(x, axis=0)
    #图像预处理
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
