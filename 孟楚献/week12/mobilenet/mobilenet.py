import keras as K
import numpy as np
from keras import Model, Input
from keras import backend as K
from keras.layers import Conv2D, BatchNormalization, Activation, DepthwiseConv2D, GlobalAveragePooling2D, Reshape, \
    Dropout
import keras.preprocessing.image as image
from keras_applications.imagenet_utils import decode_predictions

def conv_block(inputs, filters, kernel_size=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False, name="conv1")(inputs)
    x = BatchNormalization(name="conv1_bn")(x)
    x = Activation(lambda x : K.relu(x, max_value=6))(x)
    return x

def depthwise_conv_block(inputs, filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    id = str(block_id)
    x = DepthwiseConv2D((3, 3), strides, "same", depth_multiplier, use_bias=False, name="conv_dw_"+id)(inputs)
    x = BatchNormalization(name="conv_dw_bn_"+id)(x)
    x = Activation(lambda x : K.relu(x, max_value=6), name="conv_dw_relu"+id)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding="same", use_bias=False, name="conv_pw_"+id)(x)
    x = BatchNormalization(name="conv_pw_bn"+id)(x)
    x = Activation(lambda x : K.relu(x, max_value=6), name="conv_pw_relu"+id)(x)
    return x

# 深度可分离卷积神经网络
def MobileNet(input_shape=[224, 224, 3], classes=1000, depth_multiplier=1, droup_out=1e-3):
    img_input = Input(input_shape)

    # 224 x 224 x 3 -> 112 x 112 x 32
    x = conv_block(img_input, 32, strides=(2, 2))

    # 112 x 112 x 32 -> 112 x 112 x 64
    x = depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    # 112 x 112 x 64 -> 56 x 56 x 128
    x = depthwise_conv_block(x, 128, depth_multiplier, (2, 2), 2)

    # 56 x 56 x 128 -> 56 x 56 x 128
    x = depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # 56 x 56 x 128 -> 28 x 28 x 256
    x = depthwise_conv_block(x, 256, depth_multiplier, (2, 2), block_id=4)

    # 28,28,256 -> 28,28,256
    x = depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # 28,28,256 -> 14,14,512
    x = depthwise_conv_block(x, 512, depth_multiplier, (2, 2), block_id=6)

    # 14,14,512 -> 14,14,512
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 14,14,512 -> 7,7,1024
    x = depthwise_conv_block(x, 1024, depth_multiplier, (2, 2), block_id=12)
    x = depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name="reshaped_1")(x)
    x = Dropout(droup_out, name="droup_out")(x)
    x = Conv2D(classes, (1, 1), padding="same", name="conv_preds")(x)
    x = Activation("softmax", name="act_softmax")(x)
    x = Reshape((classes, ), name="reshaped_2")(x)

    inputs = img_input

    model = Model(inputs, x, name="mobilenet")
    model.load_weights("mobilenet_1_0_224_tf.h5")
    return model

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == "__main__":
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