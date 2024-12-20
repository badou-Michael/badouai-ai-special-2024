# -------------------------------------------------------------#
#   ResNet50的网络部分
# -------------------------------------------------------------#
from __future__ import print_function

import numpy as np
from keras import layers

from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model

from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


# 完全不变，输入和输出一致，能和自身串联
def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    identity_block_x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    identity_block_x = BatchNormalization(name=bn_name_base + '2a')(identity_block_x)
    identity_block_x = Activation('relu')(identity_block_x)

    identity_block_x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(identity_block_x)

    identity_block_x = BatchNormalization(name=bn_name_base + '2b')(identity_block_x)
    identity_block_x = Activation('relu')(identity_block_x)

    identity_block_x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(identity_block_x)
    identity_block_x = BatchNormalization(name=bn_name_base + '2c')(identity_block_x)

    identity_block_x = layers.add([identity_block_x, input_tensor])
    identity_block_x = Activation('relu')(identity_block_x)
    return identity_block_x


# 改变尺寸及通道数
# (输入，尺寸，权重，命名参数，命名参数)
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 卷积
    conv_block_x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    # 归一化
    conv_block_x = BatchNormalization(name=bn_name_base + '2a')(conv_block_x)
    # 激活函数
    conv_block_x = Activation('relu')(conv_block_x)

    conv_block_x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(conv_block_x)
    conv_block_x = BatchNormalization(name=bn_name_base + '2b')(conv_block_x)
    conv_block_x = Activation('relu')(conv_block_x)

    conv_block_x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(conv_block_x)
    conv_block_x = BatchNormalization(name=bn_name_base + '2c')(conv_block_x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    # 两条路结果相加
    conv_block_x = layers.add([conv_block_x, shortcut])
    conv_block_x = Activation('relu')(conv_block_x)
    return conv_block_x


def ResNet50(input_shape=[224, 224, 3], classes=1000):
    img_input = Input(shape=input_shape)
    res_net_x = ZeroPadding2D((3, 3))(img_input)

    res_net_x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(res_net_x)
    res_net_x = BatchNormalization(name='bn_conv1')(res_net_x)
    res_net_x = Activation('relu')(res_net_x)
    res_net_x = MaxPooling2D((3, 3), strides=(2, 2))(res_net_x)

    res_net_x = conv_block(res_net_x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    res_net_x = identity_block(res_net_x, 3, [64, 64, 256], stage=2, block='b')
    res_net_x = identity_block(res_net_x, 3, [64, 64, 256], stage=2, block='c')

    res_net_x = conv_block(res_net_x, 3, [128, 128, 512], stage=3, block='a')
    res_net_x = identity_block(res_net_x, 3, [128, 128, 512], stage=3, block='b')
    res_net_x = identity_block(res_net_x, 3, [128, 128, 512], stage=3, block='c')
    res_net_x = identity_block(res_net_x, 3, [128, 128, 512], stage=3, block='d')

    res_net_x = conv_block(res_net_x, 3, [256, 256, 1024], stage=4, block='a')
    res_net_x = identity_block(res_net_x, 3, [256, 256, 1024], stage=4, block='b')
    res_net_x = identity_block(res_net_x, 3, [256, 256, 1024], stage=4, block='c')
    res_net_x = identity_block(res_net_x, 3, [256, 256, 1024], stage=4, block='d')
    res_net_x = identity_block(res_net_x, 3, [256, 256, 1024], stage=4, block='e')
    res_net_x = identity_block(res_net_x, 3, [256, 256, 1024], stage=4, block='f')

    res_net_x = conv_block(res_net_x, 3, [512, 512, 2048], stage=5, block='a')
    res_net_x = identity_block(res_net_x, 3, [512, 512, 2048], stage=5, block='b')
    res_net_x = identity_block(res_net_x, 3, [512, 512, 2048], stage=5, block='c')
    # 池化
    res_net_x = AveragePooling2D((7, 7), name='avg_pool')(res_net_x)

    # 高维数据转换为一维向量
    res_net_x = Flatten()(res_net_x)
    # 全连接
    res_net_x = Dense(classes, activation='softmax', name='fc1000')(res_net_x)

    model = Model(img_input, res_net_x, name='resnet50')

    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model


if __name__ == '__main__':
    model = ResNet50()
    # 模型结构打印
    # model.summary()
    # img_path = 'elephant.jpg'
    img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # 归一化
    x = preprocess_input(x)

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
