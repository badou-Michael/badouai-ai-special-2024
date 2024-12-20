import warnings
import numpy as np

from keras.preprocessing import image

from keras.models import Model
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


def MobileNet(input_shape, depth_multiplier=1, dropout=1e-3, classes=1000):
    img_input = Input(shape=input_shape)

    # first laye
    x = Conv2d_BN(img_input, 32, (3, 3), strides=(2, 2), padding='same')
    x = depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    x = depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)
    x = depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    x = depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)
    x = depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    x = depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    x = depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    x = depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(dropout)(x)
    x = Conv2D(classes, (1, 1), padding='same')(x)
    x = Activation('softmax', name='predictions')(x)
    x = Reshape((classes,), name='reshape_2')(x)
    model = Model(img_input, x, name='mobilenet')
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name, by_name=True)
    return model


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, use_bias=False, strides=strides, name=conv_name)(x)
    x = BatchNormalization(scale=False, name=bn_name)
    x = Activation('relu', name=name)(x)
    return x


def depthwise_conv_block(inputs, pointwise_conv_filters, strides=(1, 1), block_id=1):
    x = DepthwiseConv2D((3, 3), padding='same', strides=strides, use_bias=False, name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation('relu', name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = Activation('relu', name='conv_pw_%d_relu' % block_id)(x)
    return x


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = MobileNet(input_shape=(299, 299, 3), class_num=1000)
    model.load_weights('inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
    image_path = 'elephant.jpg'
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
