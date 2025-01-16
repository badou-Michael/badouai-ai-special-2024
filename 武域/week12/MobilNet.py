import numpy as np

from keras.preprocessing import image
from keras.models import Model
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


def relu6(x):
    return K.relu(x, max_value=6)


def conv_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel_size,
               strides=strides,
               padding='same',
               use_bias=False,
               name='Conv1')(inputs)
    x = BatchNormalization(name='Conv1_bn')(x)
    x = Activation(relu6, name='Conv1_relu')(x)
    return x


def depthwise_conv_block(inputs, filters, depth_multiplier=1,
                        strides=(1, 1), block_id=1):
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)
    x = Conv2D(filters, kernel_size=(1, 1),
               strides=(1, 1),
               padding='same',
               use_bias=False,
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)
    return x


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def mobilenet(input_shape=(224, 224, 3), depth_multiplier=1, dropout=1e-3, classes=1000):
    input = Input(shape=input_shape)
    x = conv_block(input, 32, strides=(2, 2))
    x = depthwise_conv_block(x, 64, depth_multiplier=depth_multiplier, block_id=1)
    x = depthwise_conv_block(x, 128, depth_multiplier=depth_multiplier, strides=(2, 2), block_id=2)
    x = depthwise_conv_block(x, 128, depth_multiplier=depth_multiplier, block_id=3)
    x = depthwise_conv_block(x, 256, depth_multiplier=depth_multiplier, strides=(2, 2), block_id=4)
    x = depthwise_conv_block(x, 256, depth_multiplier=depth_multiplier, block_id=5)
    x = depthwise_conv_block(x, 512, depth_multiplier=depth_multiplier, strides=(2, 2), block_id=6)
    x = depthwise_conv_block(x, 512, depth_multiplier=depth_multiplier, block_id=7)
    x = depthwise_conv_block(x, 512, depth_multiplier=depth_multiplier, block_id=8)
    x = depthwise_conv_block(x, 512, depth_multiplier=depth_multiplier, block_id=9)
    x = depthwise_conv_block(x, 512, depth_multiplier=depth_multiplier, block_id=10)
    x = depthwise_conv_block(x, 512, depth_multiplier=depth_multiplier, block_id=11)
    x = depthwise_conv_block(x, 1024, depth_multiplier=depth_multiplier, strides=(2, 2), block_id=12)
    x = depthwise_conv_block(x, 1024, depth_multiplier=depth_multiplier, block_id=13)

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name="reshape_1")(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', activation='softmax', name='conv_prediction')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    inputs = input
    model = Model(inputs=inputs, outputs=x, name='MobileNet')
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)
    return model

if __name__ == '__main__':
    model = mobilenet(input_shape=(224, 224, 3), depth_multiplier=1, dropout=1e-3, classes=1000)
    img_dir = 'elephant.jpg'
    img = image.load_img(img_dir, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print(decode_predictions(preds, 1))