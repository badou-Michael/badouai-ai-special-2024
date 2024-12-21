#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/12/21 12:22
# @Author: Gift
# @File  : inception_v3.py 
# @IDE   : PyCharm
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
def conv2d_bn(x,  # 输入的张量，代表要进行操作的数据，通常是前面网络层的输出
              filters,  # 卷积操作中卷积核的数量，也就是卷积后输出特征图的通道数
              num_row,  # 卷积核的行数，用于指定卷积核在垂直方向上的尺寸大小
              num_col,  # 卷积核的列数，用于指定卷积核在水平方向上的尺寸大小
              strides=(1, 1),  # 卷积操作的步长，以二元组形式表示水平和垂直方向的步长，默认为(1, 1)，即每次移动一个像素
              padding='same',  # 卷积操作的填充方式，'same'表示在输入特征图周围进行填充，使得输出特征图尺寸与输入相同（可能会在边缘补零）；'valid'表示不进行填充
              name=None):  # 可选的名称参数，用于给创建的卷积层和批归一化层命名，方便后续查看模型结构和调试，如果为None则不特别命名
    if name is not None:
        bn_name = name + '_bn'  # 如果传入了名称，为批归一化层构建一个带后缀'_bn'的名称，便于区分不同层
        conv_name = name + '_conv'  # 为卷积层构建一个带后缀'_conv'的名称
    else:
        bn_name = None
        conv_name = None
    # 使用Conv2D层进行二维卷积操作，设置不使用偏置（因为后续的批归一化操作会对数据进行归一化处理，偏置可在归一化后再考虑）
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    # 进行批归一化操作，这里设置scale=False，表示不进行额外的缩放操作（按照批归一化的默认方式处理数据，通常是对数据进行归一化使其均值接近0，方差接近1）
    x = BatchNormalization(scale=False, name=bn_name)(x)
    # 应用ReLU激活函数，将线性的卷积输出转换为非线性输出，增强网络的表达能力，激活函数层的名称使用传入的name参数
    x = Activation('relu', name=name)(x)
    return x
def InceptionV3(input_shape=[299, 299, 3],  # 定义函数InceptionV3，设置输入图像形状的默认值，分别对应高、宽、通道数（如RGB图像通道数为3）
                classes=1000):  # 设置模型预测的类别数量默认值，用于后续全连接层输出分类结果

    img_input = Input(shape=input_shape)  # 创建模型的输入层，指定输入数据的形状

    # 以下是一系列卷积、池化操作，对输入图像进行初步特征提取和下采样
    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')  # 进行卷积、批归一化、激活操作，卷积核3x3，步长2x2，不填充边缘，输出通道数32
    x = conv2d_bn(x, 32, 3, 3, padding='valid')  # 再次类似操作，进一步提取特征
    x = conv2d_bn(x, 64, 3, 3)  # 卷积核3x3，步长(1, 1)，调整特征
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 最大池化，减小特征图尺寸

    x = conv2d_bn(x, 80, 1, 1, padding='valid')  # 1x1卷积，调整通道数等
    x = conv2d_bn(x, 192, 3, 3, padding='valid')  # 3x3卷积，提取特征
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 再次最大池化下采样

    # --------------------------------#
    #   Block1 35x35
    # --------------------------------#
    # Block1 part1
    # 35 x 35 x 192 -> 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)  # 1x1卷积分支，调整通道数
    branch5x5 = conv2d_bn(x, 48, 1, 1)  # 先1x1卷积调整通道，再5x5卷积提取特征
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)  # 多次3x3卷积，增加特征复杂度
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)  # 平均池化分支
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)  # 池化后1x1卷积调整通道
    # 拼接各分支结果，融合不同尺度特征，输出特征图尺寸为35x35x256
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed0')

    # Block1 part2
    # 35 x 35 x 256 -> 35 x 35 x 288
    # 以下类似part1结构，各分支操作后拼接，输出尺寸变为35x35x288
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed1')

    # Block1 part3
    # 35 x 35 x 288 -> 35 x 35 x 288
    # 同样结构，特征拼接后尺寸保持35x35x288
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed2')

    # --------------------------------#
    #   Block2 17x17
    # --------------------------------#
    # Block2 part1
    # 35 x 35 x 288 -> 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')  # 3x3卷积下采样，改变特征图尺寸
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)  # 多个卷积操作，部分下采样
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')
    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)  # 最大池化下采样
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')  # 拼接各分支，输出特征图尺寸为17x17x768

    # Block2 part2
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)  # 各分支不同卷积操作，提取特征
    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed4')  # 拼接各分支，尺寸保持17x17x768

    # Block2 part3 and part4
    # 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
    for i in range(2):  # 循环两次，重复类似操作，进一步挖掘特征
        branch1x1 = conv2d_bn(x, 192, 1, 1)
        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed' + str(5 + i))

    # Block2 part5
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)  # 各分支操作后拼接，尺寸保持17x17x768
    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed7')

    # --------------------------------#
    #   Block3 8x8
    # --------------------------------#
    # Block3 part1
    # 17 x 17 x 768 -> 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)  # 多个分支不同操作，下采样并调整特征，输出尺寸变为8x8x1280
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')
    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')
    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

    # Block3 part2 part3
    # 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):  # 循环两次，进行不同卷积、拼接操作，挖掘特征
        branch1x1 = conv2d_bn(x, 320, 1, 1)
        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))
        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=3)
        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed' + str(9 + i))
    # 平均池化后全连接，将特征转换为类别概率输出
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input

    model = Model(inputs, x, name='inception_v3')  # 创建模型实例，指定输入和输出

    return model
def preprocess_input(x): #简单处理图像
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
if __name__ == '__main__':
    model = InceptionV3()

    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))

