# -*- coding: utf-8 -*-
# time: 2024/11/20 21:59
# file: Inception_V3.py
# author: flame
import numpy as np
from keras import layers, Model
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image


''' 定义一个卷积层后接批量归一化和激活函数的组合函数，用于构建深度学习模型。该函数接受输入张量 x 和卷积层的参数，返回经过卷积、批量归一化和激活后的输出张量。 '''
def conv2d_bn(x, filters, num_row, num_col, strides=(1, 1), padding='same', name=None):
    ''' 检查名称是否提供，如果提供则生成批量归一化和卷积层的名称，否则设置为 None。 '''
    if name is not None:
        ''' 生成批量归一化的名称。 '''
        bn_name = name + '_bn'
        ''' 生成卷积层的名称。 '''
        conv_name = name + '_conv'
    else:
        ''' 设置批量归一化名称为 None。 '''
        bn_name = None
        ''' 设置卷积层名称为 None。 '''
        conv_name = None

    ''' 使用 Conv2D 函数创建卷积层，指定滤波器数量、卷积核大小、步长、填充方式、是否使用偏置项和名称。 '''
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False, name=conv_name)(x)

    ''' 使用 BatchNormalization 函数创建批量归一化层，指定是否使用缩放因子和名称。 '''
    x = BatchNormalization(scale=False, name=bn_name)(x)

    ''' 使用 Activation 函数创建激活层，指定激活函数类型和名称。 '''
    x = Activation('relu', name=name)(x)

    ''' 返回经过卷积、批量归一化和激活后的输出张量。 '''
    return x

''' 构建 InceptionV3 模型，该模型是一种深度卷积神经网络，用于图像分类任务。输入图像尺寸默认为 (299, 299, 3)，输出类别数默认为 1000。 '''

def inceptionV3(input_shape=(299,299,3), classes=1000):
    ''' 定义输入层，输入图像的形状为 input_shape。 '''
    img_input = Input(shape=input_shape)

    ''' 第一层卷积，使用 32 个 3x3 的卷积核，步长为 2，不使用填充。 '''
    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')

    ''' 第二层卷积，使用 32 个 3x3 的卷积核，不使用填充。 '''
    x = conv2d_bn(x, 32, 3, 3, padding='valid')

    ''' 第三层卷积，使用 64 个 3x3 的卷积核，使用默认填充。 '''
    x = conv2d_bn(x, 64, 3, 3)

    ''' 最大池化层，使用 3x3 的池化窗口，步长为 2。 '''
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    ''' 第四层卷积，使用 80 个 1x1 的卷积核，不使用填充。 '''
    x = conv2d_bn(x, 80, 1, 1, padding='valid')

    ''' 第五层卷积，使用 192 个 3x3 的卷积核，不使用填充。 '''
    x = conv2d_bn(x, 192, 3, 3, padding='valid')

    ''' 最大池化层，使用 3x3 的池化窗口，步长为 2。 '''
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    ''' 第一个 Inception 模块，包含 1x1、5x5 和 3x3 双分支卷积，以及平均池化分支。 '''
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)

    ''' 将四个分支的输出拼接在一起，形成第一个 Inception 模块的输出。 '''
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed0')

    ''' 第二个 Inception 模块，结构与第一个类似。 '''
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    ''' 将四个分支的输出拼接在一起，形成第二个 Inception 模块的输出。 '''
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed1')

    ''' 第三个 Inception 模块，结构与前两个类似。 '''
    branch1x1 = conv2d_bn(x, 64, 1, 1)
    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)

    ''' 将四个分支的输出拼接在一起，形成第三个 Inception 模块的输出。 '''
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name='mixed2')

    ''' 第四个 Inception 模块，包含 3x3 卷积和 3x3 双分支卷积，以及最大池化分支。 '''
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')
    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')
    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)

    ''' 将四个分支的输出拼接在一起，形成第四个 Inception 模块的输出。 '''
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')

    ''' 第五个 Inception 模块，包含 1x1、7x7 和 7x7 双分支卷积，以及平均池化分支。 '''
    branch1x1 = conv2d_bn(x, 192, 1, 1)
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

    ''' 将四个分支的输出拼接在一起，形成第五个 Inception 模块的输出。 '''
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed4')

    ''' 第六个到第八个 Inception 模块，结构与第五个类似。 '''
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)
        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        ''' 将四个分支的输出拼接在一起，形成第六个到第八个 Inception 模块的输出。 '''
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed' + str(5 + i))

    ''' 第九个 Inception 模块，结构与第六个到第八个类似。 '''
    branch1x1 = conv2d_bn(x, 192, 1, 1)
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

    ''' 将四个分支的输出拼接在一起，形成第九个 Inception 模块的输出。 '''
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name='mixed7')

    ''' 第十个 Inception 模块，包含 3x3 卷积和 7x7 双分支卷积，以及最大池化分支。 '''
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')
    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')
    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)

    ''' 将四个分支的输出拼接在一起，形成第十个 Inception 模块的输出。 '''
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

    ''' 第十一个到第十二个 Inception 模块，结构与第十个类似。 '''
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)

        ''' 将两个 3x3 卷积分支的输出拼接在一起。 '''
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)

        ''' 将两个 3x3 双分支卷积的输出拼接在一起。 '''
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3, name='mixed10_' + str(i))

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        ''' 将四个分支的输出拼接在一起，形成第十一个到第十二个 Inception 模块的输出。 '''
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed' + str(9 + i))

    ''' 全局平均池化层，将特征图压缩为固定大小的向量。 '''
    x = GlobalAveragePooling2D(name='avg_pool')(x)

    ''' 输出层，使用 softmax 激活函数进行多分类。 '''
    x = Dense(classes, activation='softmax', name='predictions')(x)

    ''' 定义模型的输入和输出。 '''
    inputs = img_input
    model = Model(inputs, x, name='inceptionV3')
    return model

''' 整体逻辑：此脚本加载 InceptionV3 模型，检查权重文件是否存在，加载权重，读取并预处理图像，最后进行预测并输出结果。 '''

''' 定义预处理输入数据的函数。 '''
def preprocess_input(x):
    ''' 对输入数据进行归一化处理，使其范围在 [-1, 1] 之间。 '''
    ''' 归一化输入数据至 [0, 1] 区间。 '''
    x = x / 255
    ''' 将数据均值移到 0 附近。 '''
    x = x - 0.5
    ''' 将数据标准差调整到接近 1。 '''
    x = x * 2.
    ''' 返回预处理后的输入数据。 '''
    return x

if __name__ == '__main__':
    ''' 初始化 InceptionV3 模型。 '''
    model = inceptionV3()

    ''' 检查权重文件是否存在。 '''
    import os
    file_path = 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
    ''' 如果文件路径不存在，打印错误信息。 '''
    if not os.path.exists(file_path):
       print("文件路径不存在，请检查路径是否正确。")

    ''' 加载模型权重。 '''
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    ''' 读取并调整图像大小。 '''
    img = image.load_img('bike.jpg', target_size=(299, 299))
    ''' 将图像转换为数组。 '''
    x = image.img_to_array(img)
    ''' 增加一个维度，使其符合模型输入要求。 '''
    x = np.expand_dims(x, axis=0)

    ''' 调用预处理函数处理输入数据。 '''
    x = preprocess_input(x)

    ''' 使用模型进行预测。 '''
    preds = model.predict(x)
    ''' 输出预测结果。 '''
    print('Predicted:', decode_predictions(preds))
