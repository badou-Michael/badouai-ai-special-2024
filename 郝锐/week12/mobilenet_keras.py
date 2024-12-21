#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/12/21 12:38
# @Author: Gift
# @File  : mobilenet_keras.py 
# @IDE   : PyCharm
import keras.src.activations.activations
import numpy as np

from keras.preprocessing import image

from keras.models import Model
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K

# 定义MobileNet函数，用于构建MobileNet深度学习模型结构
def MobileNet(input_shape=[224, 224, 3],  # 设置输入图像形状的默认值，分别对应高、宽、通道数（常见彩色图像通道数为3）
              depth_multiplier=1,  # 深度乘数，用于控制深度可分离卷积中深度方向上的扩展程度
              dropout=1e-3,  # 设置Dropout层的丢弃概率，用于防止过拟合
              classes=1000):  # 设置模型预测的类别数量默认值，用于后续全连接层输出分类结果

    img_input = Input(shape=input_shape)  # 创建模型的输入层，指定输入数据的形状

    # 以下是一系列卷积、深度可分离卷积等操作，对输入图像进行特征提取和下采样，逐步改变特征图的尺寸和通道数
    # 224,224,3 -> 112,112,32，进行普通卷积操作，步长为(2, 2)，用于初步下采样和特征提取
    x = _conv_block(img_input, 32, strides=(2, 2))

    # 112,112,32 -> 112,112,64，进行深度可分离卷积操作，第一个深度可分离卷积模块
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    # 112,112,64 -> 56,56,128，深度可分离卷积模块，步长为(2, 2)，进行下采样
    x = _depthwise_conv_block(x, 128, depth_multiplier,
                              strides=(2, 2), block_id=2)
    # 56,56,128 -> 56,56,128，深度可分离卷积模块，保持特征图尺寸不变，进一步提取特征
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # 56,56,128 -> 28,28,256，深度可分离卷积模块，步长为(2, 2)，再次下采样
    x = _depthwise_conv_block(x, 256, depth_multiplier,
                              strides=(2, 2), block_id=4)
    # 28,28,256 -> 28,28,256，深度可分离卷积模块，保持特征图尺寸，继续提取特征
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # 28,28,256 -> 14,14,512，深度可分离卷积模块，步长为(2, 2)，下采样
    x = _depthwise_conv_block(x, 512, depth_multiplier,
                              strides=(2, 2), block_id=6)
    # 14,14,512 -> 14,14,512，深度可分离卷积模块，保持特征图尺寸，提取特征
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 14,14,512 -> 7,7,1024，深度可分离卷积模块，步长为(2, 2)，下采样
    x = _depthwise_conv_block(x, 1024, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 7,7,1024 -> 1,1,1024，进行全局平均池化，将特征图尺寸变为1x1，对整个空间维度进行平均池化
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)  # 改变张量形状，便于后续操作
    x = Dropout(dropout, name='dropout')(x)  # 使用Dropout层，按设定概率随机丢弃神经元，防止过拟合
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)  # 1x1卷积操作，用于调整通道数，输出对应类别数量的通道
    x = Activation('softmax', name='act_softmax')(x)  # 使用softmax激活函数，将输出转换为各类别的概率分布
    x = Reshape((classes,), name='reshape_2')(x)  # 再次改变张量形状，使其维度与类别数量对应

    inputs = img_input

    model = Model(inputs, x, name='mobilenet_1_0_224_tf')  # 创建模型实例，指定输入和输出，并命名模型
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)  # 加载预训练的模型权重

    return model

# 定义_conv_block函数，用于构建一个包含卷积、批归一化和激活函数（relu6）的基本卷积模块
def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel,  # 使用Conv2D进行卷积操作，指定卷积核数量、尺寸和步长等参数
               padding='same',  # 填充方式为'same'，保持输出特征图尺寸与输入大致相同（可能会进行边缘填充）
               use_bias=False,  # 不使用偏置，因为后续批归一化操作会处理相关内容
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)  # 进行批归一化操作，对数据进行归一化处理
    return Activation(relu6, name='conv1_relu')(x)  # 使用自定义的relu6激活函数进行激活

# 定义_depthwise_conv_block函数，用于构建深度可分离卷积模块，包含深度卷积、批归一化、激活以及点卷积等操作
def _depthwise_conv_block(inputs, pointwise_conv_filters,  # 点卷积的输出通道数，即经过深度可分离卷积后调整的通道数
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    x = DepthwiseConv2D((3, 3),  # 使用深度可分离卷积，指定卷积核尺寸
                        padding='same',
                        depth_multiplier=depth_multiplier,  # 深度乘数，控制深度方向的扩展
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)  # 深度卷积后的批归一化操作
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)  # 激活操作

    x = Conv2D(pointwise_conv_filters, (1, 1),  # 进行1x1的点卷积，用于调整通道数
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)  # 点卷积后的批归一化操作
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)  # 最后激活操作

# 定义relu6函数，实现自定义的激活函数，将输入限制在最大值为6的范围内（类似ReLU，但有上限）
def relu6(x):
    return keras.src.activations.activations.ReLU(x, max_value=6)

# 定义preprocess_input函数，用于对输入图像数据进行预处理，归一化等操作
def preprocess_input(x):
    x /= 255.  # 将像素值归一化到0 - 1范围
    x -= 0.5  # 进行均值平移
    x *= 2.  # 进行缩放操作
    return x

if __name__ == '__main__':
    model = MobileNet(input_shape=(224, 224, 3))  # 创建MobileNet模型实例，使用默认的输入形状

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))  # 加载指定路径的图像，并调整到指定大小
    x = image.img_to_array(img)  # 将图像转换为数组形式
    x = np.expand_dims(x, axis=0)  # 增加一个维度，用于符合模型输入的批次维度要求
    x = preprocess_input(x)  # 对输入图像数据进行预处理
    print('Input image shape:', x.shape)  # 打印输入图像数据的形状

    preds = model.predict(x)  # 使用模型进行预测，得到预测结果
    print(np.argmax(preds))  # 打印预测结果中概率最大的类别索引
    print('Predicted:', decode_predictions(preds, 1))  # 对预测结果进行解码，只显示概率最高的一个预测类别及概率
