#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/18 21:39
@Author  : Mr.Long
@Content :
"""

from __future__ import print_function
from __future__ import absolute_import

import os.path

import numpy as np
from keras.models import Model
from keras import Input
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, \
    Dense
from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions

from common.path import inception_v3


def conv2d_bn_w12(x, filters, num_row, num_col, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn_w12'
        conv_name = name + '_conv_w12'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False, name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

def inception_v3_w12(input_shape=[299, 299, 3], classes=1000):
    img_input = Input(shape=input_shape)
    x = conv2d_bn_w12(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn_w12(x, 32, 3, 3, padding='valid')
    x = conv2d_bn_w12(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv2d_bn_w12(x, 80, 1, 1, padding='valid')
    x = conv2d_bn_w12(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    # Block1 35x35, part1:64+64+96+32 = 256
    branch_1x1 = conv2d_bn_w12(x, 64, 1, 1)
    branch_5x5 = conv2d_bn_w12(x, 48, 1, 1)
    branch_5x5 = conv2d_bn_w12(branch_5x5, 64, 5, 5)
    branch_3x3_dbl = conv2d_bn_w12(x, 64, 1, 1)
    branch_3x3_dbl = conv2d_bn_w12(branch_3x3_dbl, 96, 3, 3)
    branch_3x3_dbl = conv2d_bn_w12(branch_3x3_dbl, 96, 3, 3)
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn_w12(branch_pool, 32, 1, 1)
    x = layers.concatenate([branch_1x1, branch_5x5, branch_3x3_dbl, branch_pool], axis=3, name='mixed_0_w12')
    # Block1 35x35, part2:35 x 35 x 256 -> 35 x 35 x 288
    branch_1x1 = conv2d_bn_w12(x, 64, 1, 1)
    branch_5x5 = conv2d_bn_w12(x, 48, 1, 1)
    branch_5x5 = conv2d_bn_w12(branch_5x5, 64, 5, 5)
    branch_3x3_dbl = conv2d_bn_w12(x, 64, 1, 1)
    branch_3x3_dbl = conv2d_bn_w12(branch_3x3_dbl, 96, 3, 3)
    branch_3x3_dbl = conv2d_bn_w12(branch_3x3_dbl, 96, 3, 3)
    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn_w12(branch_pool, 64, 1, 1)
    x = layers.concatenate([branch_1x1, branch_5x5, branch_3x3_dbl, branch_pool], axis=3, name='mixed_1_w12')
    # Block1 part3:35 x 35 x 288 -> 35 x 35 x 288
    branch_1x1 = conv2d_bn_w12(x, 64, 1, 1)
    branch_5x5 = conv2d_bn_w12(x, 48, 1, 1)
    branch_5x5 = conv2d_bn_w12(branch_5x5, 64, 5, 5)
    branch_3x3_dbl = conv2d_bn_w12(x, 64, 1, 1)
    branch_3x3_dbl = conv2d_bn_w12(branch_3x3_dbl, 96, 3, 3)
    branch_3x3_dbl = conv2d_bn_w12(branch_3x3_dbl, 96, 3, 3)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn_w12(branch_pool, 64, 1, 1)
    x = layers.concatenate([branch_1x1, branch_5x5, branch_3x3_dbl, branch_pool], axis=3, name='mixed_2_w12')
    #   Block2 17x17,part1:35 x 35 x 288 -> 17 x 17 x 768
    branch_3x3 = conv2d_bn_w12(x, 384, 3, 3, strides=(2, 2), padding='valid')
    branch_3x3_dbl = conv2d_bn_w12(x, 64, 1, 1)
    branch_3x3_dbl = conv2d_bn_w12(branch_3x3_dbl, 96, 3, 3)
    branch_3x3_dbl = conv2d_bn_w12(branch_3x3_dbl, 96, 3, 3, strides=(2, 2), padding='valid')
    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch_3x3, branch_3x3_dbl, branch_pool], axis=3, name='mixed_3_w12')
    # Block2 part2:17 x 17 x 768 -> 17 x 17 x 768
    branch_1x1 = conv2d_bn_w12(x, 192, 1, 1)
    branch_7x7 = conv2d_bn_w12(x, 128, 1, 1)
    branch_7x7 = conv2d_bn_w12(branch_7x7, 128, 1, 7)
    branch_7x7 = conv2d_bn_w12(branch_7x7, 192, 7, 1)
    branch_7x7_dbl = conv2d_bn_w12(x, 128, 1, 1)
    branch_7x7_dbl = conv2d_bn_w12(branch_7x7_dbl, 128, 7, 1)
    branch_7x7_dbl = conv2d_bn_w12(branch_7x7_dbl, 128, 1, 7)
    branch_7x7_dbl = conv2d_bn_w12(branch_7x7_dbl, 128, 7, 1)
    branch_7x7_dbl = conv2d_bn_w12(branch_7x7_dbl, 192, 1, 7)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn_w12(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch_1x1, branch_7x7, branch_7x7_dbl, branch_pool], axis=3, name='mixed_4_w12')
    # Block2 part3 and part4:17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
    for i in range(2):
        branch_1x1 = conv2d_bn_w12(x, 192, 1, 1)
        branch_7x7 = conv2d_bn_w12(x, 160, 1, 1)
        branch_7x7 = conv2d_bn_w12(branch_7x7, 160, 1, 7)
        branch_7x7 = conv2d_bn_w12(branch_7x7, 192, 7, 1)
        branch_7x7_dbl = conv2d_bn_w12(x, 160, 1, 1)
        branch_7x7_dbl = conv2d_bn_w12(branch_7x7_dbl, 160, 7, 1)
        branch_7x7_dbl = conv2d_bn_w12(branch_7x7_dbl, 160, 1, 7)
        branch_7x7_dbl = conv2d_bn_w12(branch_7x7_dbl, 160, 7, 1)
        branch_7x7_dbl = conv2d_bn_w12(branch_7x7_dbl, 192, 1, 7)
        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_w12(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch_1x1, branch_7x7, branch_7x7_dbl, branch_pool], axis=3, name='mixed_' + str(5 + i) + '_w12')
    # Block2 part5:17 x 17 x 768 -> 17 x 17 x 768
    branch_1x1 = conv2d_bn_w12(x, 192, 1, 1)
    branch_7x7 = conv2d_bn_w12(x, 192, 1, 1)
    branch_7x7 = conv2d_bn_w12(branch_7x7, 192, 1, 7)
    branch_7x7 = conv2d_bn_w12(branch_7x7, 192, 7, 1)
    branch_7x7_dbl = conv2d_bn_w12(x, 192, 1, 1)
    branch_7x7_dbl = conv2d_bn_w12(branch_7x7_dbl, 192, 7, 1)
    branch_7x7_dbl = conv2d_bn_w12(branch_7x7_dbl, 192, 1, 7)
    branch_7x7_dbl = conv2d_bn_w12(branch_7x7_dbl, 192, 7, 1)
    branch_7x7_dbl = conv2d_bn_w12(branch_7x7_dbl, 192, 1, 7)
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn_w12(branch_pool, 192, 1, 1)
    x = layers.concatenate([branch_1x1, branch_7x7, branch_7x7_dbl, branch_pool], axis=3, name='mixed_7_w12')
    #   Block3 8x8, part1:17 x 17 x 768 -> 8 x 8 x 1280
    branch_3x3 = conv2d_bn_w12(x, 192, 1, 1)
    branch_3x3 = conv2d_bn_w12(branch_3x3, 320, 3, 3, strides=(2, 2), padding='valid')
    branch_7x7x3 = conv2d_bn_w12(x, 192, 1, 1)
    branch_7x7x3 = conv2d_bn_w12(branch_7x7x3, 192, 1,7)
    branch_7x7x3 = conv2d_bn_w12(branch_7x7x3, 192, 7, 1)
    branch_7x7x3 = conv2d_bn_w12(branch_7x7x3, 192, 3,3, strides=(2, 2), padding='valid')
    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch_3x3, branch_7x7x3, branch_pool], axis=3, name='mixed_8_w12')
    # Block3 part2 part3: 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):
        branch_1x1 = conv2d_bn_w12(x, 320, 1, 1)
        branch_3x3 = conv2d_bn_w12(x, 384,1,1)
        branch_3x3_1 = conv2d_bn_w12(branch_3x3, 384, 1, 3)
        branch_3x3_2 = conv2d_bn_w12(branch_3x3, 384, 3, 1)
        branch_3x3 = layers.concatenate([branch_3x3_1, branch_3x3_2], axis=3, name='mixed_9_' + str(i) + '_w12')
        branch_3x3_dbl = conv2d_bn_w12(x, 448, 1, 1)
        branch_3x3_dbl = conv2d_bn_w12(branch_3x3_dbl, 384, 3,3)
        branch_3x3_dbl_1 = conv2d_bn_w12(branch_3x3_dbl, 384, 1, 3)
        branch_3x3_dbl_2 = conv2d_bn_w12(branch_3x3_dbl, 384, 3, 1)
        branch_3x3_dbl = layers.concatenate([branch_3x3_dbl_1, branch_3x3_dbl_2], axis=3)
        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn_w12(branch_pool, 192, 1, 1)
        x = layers.concatenate([branch_1x1, branch_3x3, branch_3x3_dbl, branch_pool], axis=3, name='mixed_' + str(9 + i) + '_w12')
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    inputs = img_input
    model = Model(inputs, x, name='inception_v3')
    return model

def preprocess_input_w12(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__':
    model = inception_v3_w12()
    model.load_weights('D:\workspace\data\inceptionV3\inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
    img = image.load_img('D:\workspace\data\inceptionV3\elephant.jpg', target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input_w12(x)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))


























