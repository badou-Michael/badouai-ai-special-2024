# -------------------------------------------------------------#
#   InceptionV3的网络部分（适用于TensorFlow 2的Keras版本）
# -------------------------------------------------------------#

from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np
import keras
from keras.models import Model
from keras import layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, concatenate
from keras.utils import get_source_inputs
from keras.utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.preprocessing import image

# 设置图像数据格式为'channels_first'
K.set_image_data_format('channels_first')

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              strides=(1, 1),
              padding='same',
              name=None):
    """
    卷积层 -> 批量归一化层 -> 激活层

    参数：
        x：输入张量
        filters：滤波器数量
        num_row：卷积核高度
        num_col：卷积核宽度
        strides：步长
        padding：填充方式
        name：层名称

    返回：
        输出张量
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=1, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

def InceptionV3(input_shape=(3, 299, 299),
                classes=1000):
    """
    构建InceptionV3模型

    参数：
        input_shape：输入图像的形状
        classes：分类数量

    返回：
        InceptionV3模型实例
    """
    img_input = Input(shape=input_shape)

    # -------------------------------------------------------------
    # 构建InceptionV3网络结构
    # -------------------------------------------------------------

    # 初始卷积层和池化层
    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 第一个Inception模块组
    # Block 1
    for i in range(3):
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
        x = concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=1)  # channels_first模式下，axis=1

    # Reduction A
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=1)

    # 第二个Inception模块组
    # Block 2
    for i in range(4):
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
        x = concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=1)

    # Reduction B
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=1)

    # 第三个Inception模块组
    # Block 3
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = concatenate([branch3x3_1, branch3x3_2], axis=1)

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=1)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=1)

    # 全局平均池化层
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    # 全连接层，输出预测结果
    x = Dense(classes, activation='softmax', name='predictions')(x)

    # 创建模型
    model = Model(img_input, x, name='inception_v3')

    return model

if __name__ == '__main__':
    # 创建InceptionV3模型
    model = InceptionV3()

    # 输出模型结构
    model.summary()

    # 加载预训练权重
    weights_path = './Course_CV/Week12/demo/inceptionV3_tf/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path)

    # 加载并预处理图像
    img_path = './Course_CV/Week12/demo/inceptionV3_tf/elephant.jpg'  # 图像路径
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)

    # 将图像数据转换为符合模型的数据格式
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, 299, 299))
    else:
        x = x.reshape((299, 299, 3))

    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('输入图像的形状:', x.shape)

    # 进行预测
    preds = model.predict(x)
    print('预测结果:', decode_predictions(preds, top=5)[0])