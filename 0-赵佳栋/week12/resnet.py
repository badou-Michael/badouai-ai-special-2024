#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：resnet.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/12/18 12:25
'''


from __future__ import print_function

import numpy as np
from keras import layers

from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model

from keras.preprocessing import image
import keras.backend as K
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    定义恒等残差块。
    参数:
    - input_tensor: 输入张量
    - kernel_size: 卷积核尺寸
    - filters: 卷积层的过滤器数量列表，包含三个元素 [filters1, filters2, filters3]
    - stage: 残差块的阶段编号，用于命名
    - block: 残差块的块编号，用于命名
    """
    filters1, filters2, filters3 = filters

    # 为卷积层和批量归一化层生成唯一的名称，以便在模型中识别
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 第一个 1x1 卷积层，用于降维
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # 主卷积层，使用指定的核大小和 same 填充
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 第二个 1x1 卷积层，用于升维
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 将输入张量和处理后的张量相加，实现残差连接
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    定义卷积残差块，与恒等残差块类似，但处理输入和输出维度不匹配的情况。
    参数:
    - input_tensor: 输入张量
    - kernel_size: 卷积核尺寸
    - filters: 卷积层的过滤器数量列表，包含三个元素 [filters1, filters2, filters3]
    - stage: 残差块的阶段编号，用于命名
    - block: 残差块的块编号，用于命名
    - strides: 卷积层的步长，默认为 (2, 2)
    """
    filters1, filters2, filters3 = filters

    # 为卷积层和批量归一化层生成唯一的名称，以便在模型中识别
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 第一个 1x1 卷积层，用于降维，使用指定的步长
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # 主卷积层，使用指定的核大小和 same 填充
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 第二个 1x1 卷积层，用于升维
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 对输入张量进行 1x1 卷积，以匹配输出维度
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    # 将处理后的张量和调整维度后的输入张量相加，实现残差连接
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=[224, 224, 3], classes=1000):
    """
    构建 ResNet50 模型。
    参数:
    - input_shape: 输入图像的形状，默认为 [224, 224, 3]
    - classes: 分类的类别数，默认为 1000
    """
    # 定义输入张量
    img_input = Input(shape=input_shape)
    # 对输入图像进行零填充
    x = ZeroPadding2D((3, 3))(img_input)

    # 第一个卷积层，使用 7x7 卷积核，步长为 2
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    # 最大池化层，步长为 2
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 第二阶段，使用一个卷积残差块和两个恒等残差块
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # 第三阶段，使用一个卷积残差块和三个恒等残差块
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # 第四阶段，使用一个卷积残差块和五个恒等残差块
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3,  [256, 256, 1024], stage=4, block='f')

    # 第五阶段，使用一个卷积残差块和两个恒等残差块
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # 平均池化层
    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    # 扁平化处理
    x = Flatten()(x)
    # 全连接层，使用 softmax 激活函数进行分类
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    # 构建模型，指定输入和输出
    model = Model(img_input, x, name='resnet50')

    # 加载预训练的权重
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model


if __name__ == '__main__':
    # 创建 ResNet50 模型
    model = ResNet50()
    # 打印模型结构的摘要信息
    model.summary()
    # 加载测试图像
    img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    # 将图像转换为数组
    x = image.img_to_array(img)
    # 添加批次维度
    x = np.expand_dims(x, axis=0)
    # 对输入图像进行预处理
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    # 使用模型进行预测
    preds = model.predict(x)
    # 解码预测结果
    print('Predicted:', decode_predictions(preds))