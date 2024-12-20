# -------------------------------------------------------------#
#   MobileNet的网络部分
# -------------------------------------------------------------#
import warnings
import numpy as np

from keras.preprocessing import image

from keras.models import Model
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


def MobileNet(input_shape=[224, 224, 3],
              depth_multiplier=1,
              dropout=1e-3,
              classes=1000):
    img_input = Input(shape=input_shape)

    # 224,224,3 -> 112,112,32
    mobile_net_x = _conv_block(img_input, 32, strides=(2, 2))

    # 112,112,32 -> 112,112,64
    mobile_net_x = _depthwise_conv_block(mobile_net_x, 64, depth_multiplier, block_id=1)

    # 112,112,64 -> 56,56,128
    mobile_net_x = _depthwise_conv_block(mobile_net_x, 128, depth_multiplier,
                                         strides=(2, 2), block_id=2)
    # 56,56,128 -> 56,56,128
    mobile_net_x = _depthwise_conv_block(mobile_net_x, 128, depth_multiplier, block_id=3)

    # 56,56,128 -> 28,28,256
    mobile_net_x = _depthwise_conv_block(mobile_net_x, 256, depth_multiplier,
                                         strides=(2, 2), block_id=4)

    # 28,28,256 -> 28,28,256
    mobile_net_x = _depthwise_conv_block(mobile_net_x, 256, depth_multiplier, block_id=5)

    # 28,28,256 -> 14,14,512
    mobile_net_x = _depthwise_conv_block(mobile_net_x, 512, depth_multiplier,
                                         strides=(2, 2), block_id=6)

    # 14,14,512 -> 14,14,512
    mobile_net_x = _depthwise_conv_block(mobile_net_x, 512, depth_multiplier, block_id=7)
    mobile_net_x = _depthwise_conv_block(mobile_net_x, 512, depth_multiplier, block_id=8)
    mobile_net_x = _depthwise_conv_block(mobile_net_x, 512, depth_multiplier, block_id=9)
    mobile_net_x = _depthwise_conv_block(mobile_net_x, 512, depth_multiplier, block_id=10)
    mobile_net_x = _depthwise_conv_block(mobile_net_x, 512, depth_multiplier, block_id=11)

    # 14,14,512 -> 7,7,1024
    mobile_net_x = _depthwise_conv_block(mobile_net_x, 1024, depth_multiplier,
                                         strides=(2, 2), block_id=12)
    mobile_net_x = _depthwise_conv_block(mobile_net_x, 1024, depth_multiplier, block_id=13)

    # 7,7,1024 -> 1,1,1024
    mobile_net_x = GlobalAveragePooling2D()(mobile_net_x)
    mobile_net_x = Reshape((1, 1, 1024), name='reshape_1')(mobile_net_x)
    mobile_net_x = Dropout(dropout, name='dropout')(mobile_net_x)
    mobile_net_x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(mobile_net_x)
    mobile_net_x = Activation('softmax', name='act_softmax')(mobile_net_x)
    mobile_net_x = Reshape((classes,), name='reshape_2')(mobile_net_x)

    inputs = img_input

    model = Model(inputs, mobile_net_x, name='mobilenet_1_0_224_tf')
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)

    return model


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    conv_block_x = Conv2D(filters, kernel,
                          padding='same',
                          use_bias=False,
                          strides=strides,
                          name='conv1')(inputs)
    conv_block_x = BatchNormalization(name='conv1_bn')(conv_block_x)
    return Activation(relu6, name='conv1_relu')(conv_block_x)


def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    depthwise_conv_block_x = DepthwiseConv2D((3, 3),
                                             padding='same',
                                             depth_multiplier=depth_multiplier,
                                             strides=strides,
                                             use_bias=False,
                                             name='conv_dw_%d' % block_id)(inputs)

    depthwise_conv_block_x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(depthwise_conv_block_x)
    depthwise_conv_block_x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(depthwise_conv_block_x)

    depthwise_conv_block_x = Conv2D(pointwise_conv_filters, (1, 1),
                                    padding='same',
                                    use_bias=False,
                                    strides=(1, 1),
                                    name='conv_pw_%d' % block_id)(depthwise_conv_block_x)
    depthwise_conv_block_x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(depthwise_conv_block_x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(depthwise_conv_block_x)


# 大于6舍掉
def relu6(x):
    return K.relu(x, max_value=6)


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = MobileNet(input_shape=(224, 224, 3))

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds, 1))  # 只显示top1
