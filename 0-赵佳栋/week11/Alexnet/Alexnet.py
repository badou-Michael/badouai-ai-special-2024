#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：Alexnet.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/12/13 11:35
'''
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization



def AlexNet(input_shape=(224, 224, 3), output_shape=2):  # 图像类别为两类
    # 创建一个顺序模型对象，后续可按顺序添加网络层
    model = Sequential()

    # 第一卷积模块
    # 添加一个卷积层，使用 48 个大小为 (11, 11) 的卷积核，步长为 (4, 4)，输入形状为 input_shape，使用 valid 填充（不填充），激活函数为 relu
    # 此层用于提取图像的初始特征，通过卷积核在图像上滑动，提取不同的局部特征
    model.add(Conv2D(filters=48, kernel_size=(11, 11), strides=(4, 4), padding='valid', input_shape=input_shape,
                  activation='relu'))
    # 对卷积层的输出进行批归一化，有助于加速网络收敛和稳定训练过程
    model.add(BatchNormalization())

    # 添加一个最大池化层，池化窗口大小为 (3, 3)，步长为 (2, 2)，使用 valid 填充（不填充）
    # 池化层用于减少特征图的尺寸，保留主要特征信息，同时降低计算量
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 第 2 卷积模块
    # 再次添加一个卷积层，使用 128 个大小为 (5, 5) 的卷积核，步长为 (1, 1)，使用 same 填充（保持输入尺寸），激活函数为 relu
    model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    # 对卷积层的输出进行批归一化
    model.add(BatchNormalization())
    # 再次添加一个最大池化层，池化窗口大小为 (3, 3)，步长为 (2, 2)，使用 valid 填充（不填充）
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 继续添加卷积层，使用 192 个大小为 (3, 3) 的卷积核，步长为 (1, 1)，使用 same 填充（保持输入尺寸），激活函数为 relu
    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    # 继续添加卷积层，使用 192 个大小为 (3, 3) 的卷积核，步长为 (1, 1)，使用 same 填充（保持输入尺寸），激活函数为 relu
    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 0), padding='same', activation='relu'))
    # 继续添加卷积层，使用 128 个大小为 (3, 3) 的卷积核，步长为 (1, 1)，使用 same 填充（保持输入尺寸），激活函数为 relu
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    # 添加一个最大池化层，池化窗口大小为 (3, 3)，步长为 (2, 2)，使用 valid 填充（不填充）
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # 将池化层输出的特征图展平为一维向量，以便输入到全连接层
    model.add(Flatten())

    # 全连接层 FC1
    # 添加一个全连接层，包含 1024 个神经元，激活函数为 relu
    # 全连接层对特征进行组合和非线性变换，进一步学习特征之间的关系
    model.add(Dense(1024, activation='relu'))
    # 使用 Dropout 层，随机丢弃 25% 的神经元，防止过拟合
    model.add(Dropout(0.25))

    # 全连接层 FC2
    # 再次添加一个全连接层，包含 1024 个神经元，激活函数为 relu
    model.add(Dense(1024, activation='relu'))
    # 再次使用 Dropout 层，随机丢弃 25% 的神经元，防止过拟合
    model.add(Dropout(0.25))

    # 输出层
    # 添加一个全连接层，神经元数量为 output_shape，激活函数为 softmax，用于多分类任务，输出各类别的概率
    model.add(Dense(output_shape, activation='softmax'))

    return model