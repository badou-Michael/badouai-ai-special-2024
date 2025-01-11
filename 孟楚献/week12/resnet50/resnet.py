import keras
import numpy as np
from keras import layers, Model, Input
from keras.layers import Conv2D, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, Flatten, \
    AveragePooling2D, Dense
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input

def ResNet50(input_shape=[224, 224, 3], classes=1000):
    img_input = Input(input_shape)
    # stage 1
    x = ZeroPadding2D((3, 3))(img_input)

    x = Conv2D(64, (7, 7), strides=[2, 2], name="conv1")(x)
    x = BatchNormalization(name="bn1")(x)
    x = Activation("relu")(x)

    x = MaxPooling2D((3, 3), strides=[2, 2])(x)
    x = Activation("relu")(x)

    # stage 2
    x = conv_block(x, 3, [64, 64, 256], 2, "a", (1, 1))
    x = identity_block(x, 3, [64, 64, 256], 2, "b")
    x = identity_block(x, 3, [64, 64, 256], 2, "c")

    # stage 3
    x = conv_block(x, 3, [128, 128, 512], 3, "a")
    x = identity_block(x, 3, [128, 128, 512], 3, "b")
    x = identity_block(x, 3, [128, 128, 512], 3, "c")
    x = identity_block(x, 3, [128, 128, 512], 3, "d")

    # stage 4
    x = conv_block(x, 3, [256, 256, 1024], 4, "a")
    x = identity_block(x, 3, [256, 256, 1024], 4, "b")
    x = identity_block(x, 3, [256, 256, 1024], 4, "c")
    x = identity_block(x, 3, [256, 256, 1024], 4, "d")
    x = identity_block(x, 3, [256, 256, 1024], 4, "e")
    x = identity_block(x, 3, [256, 256, 1024], 4, "f")

    # stage 5
    x = conv_block(x, 3, [512, 512, 2048], 5, "a")
    x = identity_block(x, 3, [512, 512, 2048], 5, "b")
    x = identity_block(x, 3, [512, 512, 2048], 5, "c")

    x = AveragePooling2D((7, 7), name="average_pool")(x)

    x = Flatten()(x)
    x = Dense(classes, activation="softmax", name="fc1000")(x)

    model = Model(img_input, x, name="resnet50")
    return model

# 输入、输出维度相同，用于加深神经网络，可串联
def identity_block(input_tensor, kernel_size, filters, stage, block):
    # 三个输出通道数
    filter1, filter2, filter3 = filters

    conv_name_base = "conv_" + str(stage) + block + "_branch"
    bn_name_base = "bn_" + str(stage) + block + "_branch"

    # 1 * 1 * filter1
    x = Conv2D(filter1, (1, 1), name=conv_name_base+"2a")(input_tensor)
    x = BatchNormalization(name=bn_name_base+"2a")(x)
    x = Activation('relu')(x)

    # kernel_size * filter2
    x = Conv2D(filter2, kernel_size, padding="same", name=conv_name_base+"2b")(x)
    x = BatchNormalization(name=bn_name_base+"2b")(x)
    x = Activation("relu")(x)

    # 1 * 1 * filter3
    x = Conv2D(filter3, (1, 1), name=conv_name_base+"2c")(x)
    x = BatchNormalization(name=bn_name_base+"2c")(x)

    # short_cut
    x = layers.add([x, input_tensor])
    x = Activation("relu")(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filter1, filter2, filter3 = filters

    conv_name_base = "conv_" + str(stage) + block + "_branch"
    bn_name_base = "bn_" + str(stage) + block + "_branch"

    x = Conv2D(filter1, (1, 1), name=conv_name_base+"2a", strides=strides)(input_tensor)
    x = BatchNormalization(name=bn_name_base+"2a")(x)
    x = Activation("relu")(x)

    x = Conv2D(filter2, kernel_size, name=conv_name_base+"2b", padding='same')(x)
    x = BatchNormalization(name=bn_name_base+"2b")(x)
    x = Activation("relu")(x)

    x = Conv2D(filter3, (1, 1), name=conv_name_base+"2c")(x)
    x = BatchNormalization(name=bn_name_base+"2c")(x)

    short_cut = Conv2D(filter3, (1, 1), name=conv_name_base+"1",strides=strides)(input_tensor)
    short_cut = BatchNormalization(name=bn_name_base+"1")(short_cut)

    x = layers.add([x, short_cut])
    x = Activation("relu")(x)
    return x

if __name__ == "__main__":
    resnet_model = ResNet50()
    resnet_model.load_weights("../参考/resnet50_tf/resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    resnet_model.summary()

    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('Input image shape:', x.shape)
    preds = resnet_model.predict(x)
    print('Predicted:', decode_predictions(preds))