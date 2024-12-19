from __future__ import print_function

import numpy as np
from keras import layers

from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model

from keras.preprocessing import image
import keras.backend as K
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input


def conv_block(input_x, filters, stage, block, strides=(2, 2)):
    f1, f2, f3 = filters
    base_name = 'stage_%d_cov_block_%d_' % (stage, block)

    s = Conv2D(f1, (1, 1), strides=strides, name=f'{base_name}_conv_1')(input_x)
    s = BatchNormalization(name=f'{base_name}_conv_1_bn')(s)
    s = Activation('relu')(s)

    s = Conv2D(f2, (3, 3), padding='same', name=f'{base_name}_conv_2')(s)
    s = BatchNormalization(name=f'{base_name}_conv_2_bn')(s)
    s = Activation('relu')(s)

    s = Conv2D(f3, (1, 1), name=f'{base_name}_conv_3')(s)
    s = BatchNormalization(name=f'{base_name}_conv_3_bn')(s)

    shortcut = Conv2D(f3, (1, 1), strides=strides, name=f'{base_name}_conv_4')(input_x)
    shortcut = BatchNormalization(name=f'{base_name}_conv_4_bn')(shortcut)

    s = layers.add([s , shortcut])
    return Activation('relu')(s)


def identity_block(input_x, filters, stage, block):
    f1, f2, f3 = filters
    base_name = 'stage_%d_identity_block_%d_' % (stage, block)

    s = Conv2D(f1, (1, 1), name=f'{base_name}_conv_1')(input_x)
    s = BatchNormalization(name=f'{base_name}_conv_1_bn')(s)
    s = Activation('relu')(s)

    s = Conv2D(f2, (3, 3), padding='same', name=f'{base_name}_conv_2')(s)
    s = BatchNormalization(name=f'{base_name}_conv_2_bn')(s)
    s = Activation('relu')(s)

    s = Conv2D(f3, (1, 1), name=f'{base_name}_conv_3')(s)
    s = BatchNormalization(name=f'{base_name}_conv_3_bn')(s)

    s = layers.add([s , input_x])
    return Activation('relu')(s)


def resnet50(input_shape=[224, 224, 3]):
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(img_input)

    x = Conv2D(64, (7 , 7), strides=(2, 2),  name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, [64, 64, 256], 1, 1, strides=(1,1))
    x = identity_block(x, [64, 64, 256], 1, 2)
    x = identity_block(x, [64, 64, 256], 1, 3)

    x = conv_block(x, [128, 128, 512], 2, 1, strides=(2, 2))
    for i in range(2, 5):
        x = identity_block(x, [128, 128, 512], 2, i)

    x = conv_block(x, [256, 256, 1024], 3, 1, strides=(2, 2))
    for i in range(2, 7):
        x = identity_block(x, [256, 256, 1024], 3, i)

    x = conv_block(x, [512, 512, 2048], 4, 1)
    x = identity_block(x, [512, 512, 2048], 4, 2)
    x = identity_block(x, [512, 512, 2048], 4, 3)

    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)

    x = Dense(1000, activation='softmax')(x)

    model = Model(img_input, x, name='resnet50')

    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model


if __name__ == '__main__':
    model = resnet50()
    model.summary()
    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
