from keras.layers import Activation, DepthwiseConv2D, BatchNormalization, Conv2D, Input, GlobalAveragePooling2D, \
    Dropout, Reshape, Flatten
from keras.models import Model
from keras.preprocessing import image
from keras import backend as k
import numpy as np
from keras.applications.imagenet_utils import decode_predictions


def relu6(x):
    return k.relu(x, max_value=6)  # 正常情况下,relu 右边是无限向上的，现在限制了y值最大为6


def _depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    x = DepthwiseConv2D((3, 3),
                        strides=strides,
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        use_bias=False,
                        name='conv_dw_%d' % block_id
                        )(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same',
               use_bias=False,
               name='conv_pw_%d' % block_id
               )(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

    return x


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel_size=kernel, strides=strides, padding='same', use_bias=False, name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation(relu6, name='conv1_relu')(x)

    return x


def Mobilenet(inputs=[224, 224, 3], depth_multiplier=1, dropout=1e-3, classes=1000):
    inputs_image = Input(shape=inputs)

    # 224,224,3 -> 112,112,32
    x = _conv_block(inputs_image, 32, strides=(2, 2))
    # 112,112,32 -> 112,112,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)
    # 112,112,64 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier,
                              strides=(2, 2), block_id=2)
    # 56,56,128 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # 56,56,128 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier,
                              strides=(2, 2), block_id=4)

    # 28,28,256 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # 28,28,256 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier,
                              strides=(2, 2), block_id=6)

    # 14,14,512 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 14,14,512 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 7,7,1024 -> 1,1,1024
    # GlobalAveragePooling2D:平均池化，不需要指定pool_size 和 strides 等参数，操作的实质是将
    # 输入特征图的每一个通道求平均得到一个数值。
    x = GlobalAveragePooling2D()(x)
    # Reshape(height,weight,channel)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    # 全连接
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    model = Model(inputs_image, x, name='mobilenet_1_0_224_tf')
    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


if __name__ == '__main__':
    model = Mobilenet(inputs=(224, 224, 3))
    model.load_weights('E:/YAN/HelloWorld/cv/【12】图像识别/代码/mobilenet/mobilenet_1_0_224_tf.h5')
    img = image.load_img('E:/YAN/HelloWorld/cv/【12】图像识别/代码/mobilenet/elephant.jpg', target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # 归一化 此模块用keras.applications.imagenet_utils.preprocess_input 图片会被识别为跳水台
    print('Input image shape:', img.shape)
    preds = model.predict(img)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds, 1))  # decode_predictions=vgg6_utils.print_prob

