from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Activation, Dense, Input, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input


def conv2d_bn(input_x, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    conv_name = name if name is None else f'{name}_conv'
    bn_name = name if name is None else f'{name}_bn'

    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=conv_name)(input_x)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def inception_a(input_x, name, pooling_filters=32):
    inception_a_1x1 = conv2d_bn(input_x, 64, (1, 1))

    inception_a_5x5 = conv2d_bn(input_x, 48, (1, 1))
    inception_a_5x5 = conv2d_bn(inception_a_5x5, 64, (5, 5))

    inception_a_3x3 = conv2d_bn(input_x, 64, (1, 1))
    inception_a_3x3 = conv2d_bn(inception_a_3x3, 96, (3, 3))
    inception_a_3x3 = conv2d_bn(inception_a_3x3, 96, (3, 3))

    inception_a_pooling = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_x)
    inception_a_pooling = conv2d_bn(inception_a_pooling, pooling_filters, (1, 1))

    return layers.concatenate([inception_a_1x1, inception_a_5x5, inception_a_3x3, inception_a_pooling], axis=3,
                              name=name)


def InceptionV3(input_shape=[299, 299, 3]):
    img_input = Input(shape=input_shape)

    x = conv2d_bn(img_input, 32, (3, 3), strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, (3, 3), padding='valid')
    x = conv2d_bn(x, 64, (3, 3))
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv2d_bn(x, 80, (1, 1), padding='valid')
    x = conv2d_bn(x, 192, (3, 3), padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    '''
    InceptionA × 3
    '''
    x = inception_a(x, name='mixed0_0')
    x = inception_a(x, pooling_filters=64, name='mixed0_1')
    x = inception_a(x, pooling_filters=64, name='mixed0_2')

    '''
    InceptionB
    '''
    inception_b_3x3 = conv2d_bn(x, 384, (3, 3), strides=(2, 2), padding='valid')

    inception_b_3x3_2 = conv2d_bn(x, 64, (1, 1))
    inception_b_3x3_2 = conv2d_bn(inception_b_3x3_2, 96, (3, 3))
    inception_b_3x3_2 = conv2d_bn(inception_b_3x3_2, 96, (3, 3), strides=(2, 2), padding='valid')

    inception_b_pooling = AveragePooling2D((3, 3), strides=(2, 2))(x)

    x = layers.concatenate([inception_b_3x3, inception_b_3x3_2, inception_b_pooling], axis=3, name='mixed1')

    '''
    InceptionC
    '''
    inception_c_1x1 = conv2d_bn(x, 192, (1, 1))

    inception_c_7x7 = conv2d_bn(x, 128, (1, 1))
    inception_c_7x7 = conv2d_bn(inception_c_7x7, 128, (1, 7))
    inception_c_7x7 = conv2d_bn(inception_c_7x7, 192, (7, 1))

    inception_c_7x7_2 = conv2d_bn(x, 128, (1, 1))
    inception_c_7x7_2 = conv2d_bn(inception_c_7x7_2, 128, (7, 1))
    inception_c_7x7_2 = conv2d_bn(inception_c_7x7_2, 128, (1, 7))
    inception_c_7x7_2 = conv2d_bn(inception_c_7x7_2, 128, (7, 1))
    inception_c_7x7_2 = conv2d_bn(inception_c_7x7_2, 192, (1, 7))

    inception_c_pooling = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    inception_c_pooling = conv2d_bn(inception_c_pooling, 192, (1, 1))

    x = layers.concatenate([inception_c_1x1, inception_c_7x7, inception_c_7x7_2, inception_c_pooling], axis=3,
                           name='mixed2')

    '''
    InceptionD
    '''
    for i in range(2):
        inception_d_1x1 = conv2d_bn(x, 192, (1, 1))

        inception_d_7x7 = conv2d_bn(x, 160, (1, 1))
        inception_d_7x7 = conv2d_bn(inception_d_7x7, 160, (1, 7))
        inception_d_7x7 = conv2d_bn(inception_d_7x7, 192, (7, 1))

        inception_d_7x7_3 = conv2d_bn(x, 160, (1, 1))
        inception_d_7x7_3 = conv2d_bn(inception_d_7x7_3, 160, (7, 1))
        inception_d_7x7_3 = conv2d_bn(inception_d_7x7_3, 160, (1, 7))
        inception_d_7x7_3 = conv2d_bn(inception_d_7x7_3, 160, (7, 1))
        inception_d_7x7_3 = conv2d_bn(inception_d_7x7_3, 192, (1, 7))

        inception_d_pooling = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        inception_d_pooling = conv2d_bn(inception_d_pooling, 192, (1, 1))

        x = layers.concatenate([inception_d_1x1, inception_d_7x7, inception_d_7x7_3, inception_d_pooling], axis=3,
                               name='mixed3_%s'%i)

    '''
    InceptionE
    '''
    inception_e_1x1 = conv2d_bn(x, 192, (1, 1))

    inception_e_7x7 = conv2d_bn(x, 192, (1, 1))
    inception_e_7x7 = conv2d_bn(inception_e_7x7, 192, (1, 7))
    inception_e_7x7 = conv2d_bn(inception_e_7x7, 192, (7, 1))

    inception_e_7x7_3 = conv2d_bn(x, 192, (1, 1))
    inception_e_7x7_3 = conv2d_bn(inception_e_7x7_3, 192, (7, 1))
    inception_e_7x7_3 = conv2d_bn(inception_e_7x7_3, 192, (1, 7))
    inception_e_7x7_3 = conv2d_bn(inception_e_7x7_3, 192, (7, 1))
    inception_e_7x7_3 = conv2d_bn(inception_e_7x7_3, 192, (1, 7))

    inception_e_pooling = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    inception_e_pooling = conv2d_bn(inception_e_pooling, 192, (1, 1))

    x = layers.concatenate([inception_e_1x1, inception_e_7x7, inception_e_7x7_3, inception_e_pooling], axis=3,
                           name='mixed4')

    '''
    InceptionF
    '''
    inception_f_3x3 = conv2d_bn(x, 192, (1, 1))
    inception_f_3x3 = conv2d_bn(inception_f_3x3, 320, (3, 3), strides=(2, 2), padding='valid')

    inception_f_7x7 = conv2d_bn(x, 192, (1, 1))
    inception_f_7x7 = conv2d_bn(inception_f_7x7, 192, (1, 7))
    inception_f_7x7 = conv2d_bn(inception_f_7x7, 192, (7, 1))
    inception_f_7x7 = conv2d_bn(inception_f_7x7, 192, (3, 3), strides=(2, 2), padding='valid')

    inception_f_pooling = AveragePooling2D((3, 3), strides=(2, 2))(x)

    x = layers.concatenate([inception_f_3x3, inception_f_7x7, inception_f_pooling], axis=3, name='mixed5')

    '''
    InceptionG
    '''
    for i in range(2):
        inception_g_1x1 = conv2d_bn(x, 320, (1, 1))

        inception_g_3x3 = conv2d_bn(x, 384, (1, 1))
        inception_g_3x3_1 = conv2d_bn(inception_g_3x3, 384, (1, 3))
        inception_g_3x3_2 = conv2d_bn(inception_g_3x3, 384, (3, 1))
        inception_g_3x3 = layers.concatenate([inception_g_3x3_1, inception_g_3x3_2], axis=3, name='mixed6_%s'%i)

        inception_g_3x3d = conv2d_bn(x, 448, (1, 1))
        inception_g_3x3d = conv2d_bn(inception_g_3x3d, 384, (3, 3))
        inception_g_3x3d_1 = conv2d_bn(inception_g_3x3d, 384, (1, 3))
        inception_g_3x3d_2 = conv2d_bn(inception_g_3x3d, 384, (3, 1))
        inception_g_3x3d = layers.concatenate([inception_g_3x3d_1, inception_g_3x3d_2], axis=3, name='mixed7_%s'%i)

        inception_g_pooling = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        inception_g_pooling = conv2d_bn(inception_g_pooling, 192, (1, 1))

        x = layers.concatenate([inception_g_1x1, inception_g_3x3, inception_g_3x3d, inception_g_pooling], axis=3)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='softmax')(x)

    return Model(img_input, x)

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
