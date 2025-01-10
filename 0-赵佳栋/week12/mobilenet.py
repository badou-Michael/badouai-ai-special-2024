#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：mobilenet.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/12/19 13:25
'''
import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    """
    标准卷积块函数。
    参数:
    - inputs: 输入张量。
    - filters: 卷积层的滤波器数量。
    - kernel: 卷积核大小，默认为 (3, 3)。
    - strides: 卷积步长，默认为 (1, 1)。
    步骤:
    1. 应用卷积层，不使用偏置项。
    2. 应用批归一化。
    3. 应用 relu6 激活函数。
    """
    x = Conv2D(filters, kernel,
              padding='same',
              use_bias=False,
              strides=strides,
              name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters,
                       depth_multiplier=1, strides=(1, 1), block_id=1):
    """
    深度可分离卷积块函数。
    参数:
    - inputs: 输入张量。
    - pointwise_conv_filters: 逐点卷积的滤波器数量。
    - depth_multiplier: 深度乘法器，默认为 1。
    - strides: 深度卷积的步长，默认为 (1, 1)。
    - block_id: 块的唯一标识符，用于命名层。
    步骤:
    1. 应用深度卷积层，不使用偏置项。
    2. 应用批归一化。
    3. 应用 relu6 激活函数。
    4. 应用逐点卷积层，不使用偏置项。
    5. 应用批归一化。
    6. 应用 relu6 激活函数。
    """
    x = DepthwiseConv2D((3, 3),
                     padding='same',
                     depth_multiplier=depth_multiplier,
                     strides=strides,
                     use_bias=False,
                     name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
              padding='same',
              use_bias=False,
              strides=(1, 1),
              name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def relu6(x):
    """
    自定义的 relu6 激活函数，将输入张量中大于 6 的值截断为 6。
    """
    return K.relu(x, max_value=6)


def MobileNet(input_shape=[224, 224, 3], depth_multiplier=1, dropout=1e-3, classes=1000):
    """
    MobileNet 模型构建函数。
    参数:
    - input_shape: 输入图像的形状，默认为 [224, 224, 3]。
    - depth_multiplier: 深度乘法器，默认为 1。
    - dropout: Dropout 概率，默认为 1e-3。
    - classes: 分类的类别数，默认为 1000。
    步骤:
    1. 定义输入张量。
    2. 应用初始的标准卷积块。
    3. 应用多个深度可分离卷积块，其中部分块的步长为 2 进行下采样。
    4. 应用全局平均池化。
    5. 重塑张量。
    6. 应用 Dropout 层。
    7. 应用最终的卷积层和 softmax 激活进行分类。
    8. 再次重塑张量。
    9. 构建模型并加载预训练权重。
    """
    img_input = Input(shape=input_shape)

    x = _conv_block(img_input, 32, strides=(2, 2))

    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 128, depth_multiplier,
                           strides=(2, 2), block_id=2)

    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x, 256, depth_multiplier,
                           strides=(2, 2), block_id=4)

    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 512, depth_multiplier,
                           strides=(2, 2), block_id=6)

    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, 1024, depth_multiplier,
                           strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    inputs = img_input

    model = Model(inputs, x, name='mobilenet_1_0_224_tf')
    model.load_weights("mobilenet_1_0_224_tf.h5")

    return model


def preprocess_input(x):
    """
    输入预处理函数，将输入归一化到 [-1, 1] 范围。
    """
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    # 创建 MobileNet 模型
    model = MobileNet(input_shape=(224, 224, 3))
    # 打印模型摘要信息
    model.summary()

    img_path = 'elephant.jpg'
    # 加载图像并调整大小
    img = image.load_img(img_path, target_size=(224, 224))
    # 将图像转换为数组
    x = image.img_to_array(img)
    # 添加批次维度
    x = np.expand_dims(x, axis=0)
    # 预处理输入
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    # 使用模型进行预测
    preds = model.predict(x)
    # 输出预测结果中最大概率的类别索引
    print(np.argmax(preds))
    # 打印预测结果中最大概率的类别及其概率
    print('Predicted:', decode_predictions(preds, 1))