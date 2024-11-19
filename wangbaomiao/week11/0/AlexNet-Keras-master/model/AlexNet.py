# -*- coding: utf-8 -*-
# time: 2024/11/19 15:23
# file: AlexNet.py
# author: flame
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import Adam

''' 定义 AlexNet 模型，输入图像尺寸为 (224, 224, 3)，输出类别数为 2。模型结构包括多个卷积层、批归一化层、池化层、全连接层和 Dropout 层。 '''

def AlexNet(input_shape=(224, 224, 3), output_shape=2):
    ''' 创建一个Sequential模型，用于堆叠各个网络层。 '''
    model = Sequential()

    ''' 添加第一个卷积层，使用 28 个 11x11 的卷积核，步长为 4，激活函数为 ReLU。输入图像尺寸为 input_shape。 '''
    model.add(Conv2D(filters=28, kernel_size=(11, 11), strides=(4, 4), padding='valid', input_shape=input_shape, activation='relu'))
    ''' 添加批归一化层，用于加速训练过程并提高模型性能。 '''
    model.add(BatchNormalization())
    ''' 添加最大池化层，使用 3x3 的池化窗口，步长为 2，用于减少特征图的尺寸。 '''
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    ''' 添加第二个卷积层，使用 128 个 5x5 的卷积核，步长为 1，填充方式为 same，激活函数为 ReLU。 '''
    model.add(Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    ''' 添加批归一化层，用于加速训练过程并提高模型性能。 '''
    model.add(BatchNormalization())
    ''' 添加最大池化层，使用 3x3 的池化窗口，步长为 2，用于减少特征图的尺寸。 '''
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    ''' 添加第三个卷积层，使用 192 个 3x3 的卷积核，步长为 1，填充方式为 same，激活函数为 ReLU。 '''
    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    ''' 添加第四个卷积层，使用 192 个 3x3 的卷积核，步长为 1，填充方式为 same，激活函数为 ReLU。 '''
    model.add(Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    ''' 添加第五个卷积层，使用 128 个 3x3 的卷积核，步长为 1，填充方式为 same，激活函数为 ReLU。 '''
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    ''' 添加最大池化层，使用 3x3 的池化窗口，步长为 2，用于减少特征图的尺寸。 '''
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    ''' 将特征图展平为一维向量，以便输入到全连接层。 '''
    model.add(Flatten())
    ''' 添加第一个全连接层，包含 1024 个神经元，激活函数为 ReLU。 '''
    model.add(Dense(1024, activation='relu'))
    ''' 添加 Dropout 层，丢弃率为 0.25，用于防止过拟合。 '''
    model.add(Dropout(0.25))

    ''' 添加第二个全连接层，包含 1024 个神经元，激活函数为 ReLU。 '''
    model.add(Dense(1024, activation='relu'))
    ''' 添加 Dropout 层，丢弃率为 0.25，用于防止过拟合。 '''
    model.add(Dropout(0.25))

    ''' 添加输出层，包含 output_shape 个神经元，激活函数为 softmax，用于分类任务。 '''
    model.add(Dense(output_shape, activation='softmax'))

    ''' 返回构建好的模型。 '''
    return model
