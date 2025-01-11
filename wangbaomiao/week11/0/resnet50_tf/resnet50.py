# -*- coding: utf-8 -*-
# time: 2024/11/19 19:13
# file: resnet50.py
# author: flame
from __future__ import print_function

import h5py
import numpy as np
from keras import layers
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Activation, BatchNormalization, Flatten
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Input
from keras.models import Model
from keras.preprocessing import image


''' 
实现 ResNet50 模型，包括身份块（identity block）和卷积块（convolutional block）。模型输入为指定形状的图像，输出为分类结果。
'''

''' 定义身份块（identity block），用于 ResNet 中的残差连接。 '''
def identity_block(input_tensor, kernel_size, filters, stage, block):
    ''' 解构 filters 列表，获取三个滤波器的数量。 '''
    filters1, filters2, filters3 = filters

    ''' 构建卷积层和批量归一化层的基础名称。 '''
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    ''' 应用 1x1 卷积层，减少特征图的通道数。 '''
    X = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    ''' 应用批量归一化，加速训练过程。 '''
    X = BatchNormalization(name=bn_name_base + '2a')(X)
    ''' 应用 ReLU 激活函数，引入非线性。 '''
    X = Activation('relu')(X)

    ''' 应用 3x3 卷积层，提取特征。 '''
    X = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(X)
    ''' 应用批量归一化，加速训练过程。 '''
    X = BatchNormalization(name=bn_name_base + '2b')(X)
    ''' 应用 ReLU 激活函数，引入非线性。 '''
    X = Activation('relu')(X)

    ''' 应用 1x1 卷积层，增加特征图的通道数。 '''
    X = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(X)
    ''' 应用批量归一化，加速训练过程。 '''
    X = BatchNormalization(name=bn_name_base + '2c')(X)
    ''' 将主路径和输入张量相加，形成残差连接。 '''
    X = layers.add([X, input_tensor])
    ''' 应用 ReLU 激活函数，引入非线性。 '''
    X = Activation('relu')(X)
    ''' 返回处理后的张量。 '''
    return X

''' 定义卷积块（convolutional block），用于 ResNet 中的下采样。 '''
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    ''' 解构 filters 列表，获取三个滤波器的数量。 '''
    filters1, filters2, filters3 = filters

    ''' 构建卷积层和批量归一化层的基础名称。 '''
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    ''' 应用 1x1 卷积层，减少特征图的通道数，并进行下采样。 '''
    X = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    ''' 应用批量归一化，加速训练过程。 '''
    X = BatchNormalization(name=bn_name_base + '2a')(X)
    ''' 应用 ReLU 激活函数，引入非线性。 '''
    X = Activation('relu')(X)

    ''' 应用 3x3 卷积层，提取特征。 '''
    X = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(X)
    ''' 应用批量归一化，加速训练过程。 '''
    X = BatchNormalization(name=bn_name_base + '2b')(X)
    ''' 应用 ReLU 激活函数，引入非线性。 '''
    X = Activation('relu')(X)

    ''' 应用 1x1 卷积层，增加特征图的通道数。 '''
    X = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(X)
    ''' 应用批量归一化，加速训练过程。 '''
    X = BatchNormalization(name=bn_name_base + '2c')(X)

    ''' 应用 1x1 卷积层，调整 shortcut 路径的通道数和尺寸。 '''
    shortcut = Conv2D(filters3, [1, 1], strides=strides, name=conv_name_base + '1')(input_tensor)
    ''' 应用批量归一化，加速训练过程。 '''
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    ''' 将主路径和 shortcut 路径相加，形成残差连接。 '''
    X = layers.add([X, shortcut])
    ''' 应用 ReLU 激活函数，引入非线性。 '''
    X = Activation('relu')(X)
    ''' 返回处理后的张量。 '''
    return X

''' 定义 ResNet50 模型。 '''
def ResNet50(input_shape=(224, 224, 3), classes=1000):
    ''' 创建输入张量。 '''
    img_input = Input(shape=input_shape)

    ''' 添加零填充层，确保输入图像尺寸适合后续卷积操作。 '''
    X = ZeroPadding2D((3, 3))(img_input)
    ''' 应用 7x7 卷积层，提取初始特征。 '''
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(X)
    ''' 应用批量归一化，加速训练过程。 '''
    X = BatchNormalization(name='bn_conv1')(X)
    ''' 应用 ReLU 激活函数，引入非线性。 '''
    X = Activation('relu')(X)
    ''' 应用最大池化层，进一步下采样。 '''
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    ''' 应用第一个卷积块，不进行下采样。 '''
    X = conv_block(X, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    ''' 应用两个身份块，保持特征图尺寸不变。 '''
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    ''' 应用第二个卷积块，进行下采样。 '''
    X = conv_block(X, 3, [128, 128, 512], stage=3, block='a')
    ''' 应用三个身份块，保持特征图尺寸不变。 '''
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    ''' 应用第三个卷积块，进行下采样。 '''
    X = conv_block(X, 3, [256, 256, 1024], stage=4, block='a')
    ''' 应用五个身份块，保持特征图尺寸不变。 '''
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    ''' 应用第四个卷积块，进行下采样。 '''
    X = conv_block(X, 3, [512, 512, 2048], stage=5, block='a')
    ''' 应用两个身份块，保持特征图尺寸不变。 '''
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    ''' 应用全局平均池化层，将特征图降维。 '''
    X = AveragePooling2D((7, 7), name='avg_pool')(X)

    ''' 展平特征图，准备全连接层输入。 '''
    X = Flatten()(X)

    ''' 应用全连接层，输出分类结果。 '''
    X = Dense(classes, activation='softmax', name='fc1000')(X)

    ''' 创建模型实例。 '''
    model = Model(img_input, X, name='resnet50')
    ''' 打印模型层数。 '''
    print(f"Model layers: {len(model.layers)}")
    ''' 打开权重文件，检查层数。 '''
    with h5py.File('resnet50_weights_tf_dim_ordering_tf_kernels.h5', 'r') as f:
        print(f"Weight file layers: {len(f.keys())}")
    ''' 加载预训练权重。 '''
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    ''' 返回构建好的模型。 '''
    return model

if __name__ == '__main__':
    ''' 构建并打印 ResNet50 模型的结构。 '''
    model = ResNet50()
    model.summary()

    ''' 定义图像路径。 '''
    img_path = 'bike.jpg'
    ''' 加载并调整图像大小。 '''
    img = image.load_img(img_path, target_size=(224, 224))
    ''' 将图像转换为数组。 '''
    X = image.img_to_array(img)
    ''' 增加一个维度，使其成为批量输入。 '''
    X = np.expand_dims(X, axis=0)
    ''' 预处理图像，使其符合模型输入要求。 '''
    X = preprocess_input(X)
    ''' 打印输入图像的形状。 '''
    print('Input image shape:', X.shape)
    ''' 打印模型层数。 '''
    print(f"Model layers: {len(model.layers)}")
    ''' 进行预测。 '''
    preds = model.predict(X)
    ''' 打印预测结果。 '''
    print('Predicted:', decode_predictions(preds))

'''为什么在第3阶段增加一个身份块可以增加6层有效，而在第4阶段和第5阶段却无效？
实际上，无论是在第3阶段、第4阶段还是第5阶段增加一个身份块，都会增加相同的层数（6层）。这是因为每个身份块的结构是固定的，包含3个卷积层和3个批量归一化层。增加一个身份块总是会增加这6层。
可能的原因
权重文件不匹配：
如果你在第4阶段或第5阶段增加身份块后，尝试加载预训练的权重文件（例如 resnet50_weights_tf_dim_ordering_tf_kernels.h5），可能会遇到权重不匹配的问题。预训练的权重文件是针对特定层数的模型设计的，增加层数后，权重文件中的层名称和数量不再匹配，导致加载失败。
模型结构不一致：
如果你在第4阶段或第5阶段增加身份块，但没有相应地调整其他部分的模型结构，可能会导致模型结构不一致，从而影响模型的性能和训练效果。
解决方法
手动调整权重：
如果你需要在第4阶段或第5阶段增加身份块，可以手动调整权重文件，使其与新的模型结构匹配。这通常需要对权重文件进行编辑，将新增的层的权重初始化为合理的值。
重新训练模型：
如果你有足够的时间和计算资源，可以考虑从头开始训练模型。这样可以避免权重不匹配的问题，但需要大量的数据和计算资源。'''