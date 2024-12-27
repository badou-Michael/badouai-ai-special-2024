from keras.layers import Conv2D, BatchNormalization, Activation, Input, MaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D, Dense
from keras import layers
from keras.models import Model
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input, decode_predictions


def cov_bn(x, fileters, row, col, strides=(1, 1), padding="same"):
    x = Conv2D(fileters, kernel_size=(row, col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(scale=False)(x)
    x = Activation("relu")(x)
    return x


def inceptionV3Model(input_shape=[299, 299, 3], classes=1000):
    """
    @param input_shape:
    @param classes:
    @return:
    """
    image_input = Input(shape=input_shape)

    x = cov_bn(image_input, 32, 3, 3, strides=(2, 2), padding="valid")
    x = cov_bn(x, 32, 3, 3, strides=(1, 1), padding="valid")
    x = cov_bn(x, 64, 3, 3, strides=(1, 1))
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = cov_bn(x, 80, 1, 1, padding='valid')
    x = cov_bn(x, 192, 3, 3, padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 第一个并联开始
    s1_b1 = cov_bn(x, 64, 1, 1, strides=(1, 1), padding="same")

    s1_b2 = cov_bn(x, 48, 1, 1, strides=(1, 1), padding="same")
    s1_b2 = cov_bn(s1_b2, 64, 5, 5, strides=(1, 1), padding="same")

    s1_b3 = cov_bn(x, 64, 1, 1, strides=(1, 1), padding="same")
    s1_b3 = cov_bn(s1_b3, 96, 3, 3, strides=(1, 1), padding="same")
    s1_b3 = cov_bn(s1_b3, 96, 3, 3, strides=(1, 1), padding="same")

    s1_b4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    s1_b4 = cov_bn(s1_b4, 32, 1, 1, strides=(1, 1), padding="same")

    # 因为模式（n,h,w,c）合并通道为索引为3，所以axis=3
    x = layers.concatenate([s1_b1, s1_b2, s1_b3, s1_b4], axis=3, name="mix0")

    # 第二个并联开始
    for i in range(2):
        s2_b1 = cov_bn(x, 64, 1, 1, strides=(1, 1), padding="same")

        s2_b2 = cov_bn(x, 48, 1, 1, strides=(1, 1), padding="same")
        s2_b2 = cov_bn(s2_b2, 64, 5, 5, strides=(1, 1), padding="same")

        s2_b3 = cov_bn(x, 64, 1, 1, strides=(1, 1), padding="same")
        s2_b3 = cov_bn(s2_b3, 96, 3, 3, strides=(1, 1), padding="same")
        s2_b3 = cov_bn(s2_b3, 96, 3, 3, strides=(1, 1), padding="same")

        s2_b4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
        s2_b4 = cov_bn(s2_b4, 64, 1, 1, strides=(1, 1), padding="same")

        x = layers.concatenate([s2_b1, s2_b2, s2_b3, s2_b4], axis=3, name="mixr" + str(i))

    # 第3个并联开始
    s3_b1 = cov_bn(x, 384, 3, 3, strides=(2, 2), padding="valid")

    s3_b2 = cov_bn(x, 64, 1, 1, strides=(1, 1), padding="same")
    s3_b2 = cov_bn(s3_b2, 96, 3, 3, strides=(1, 1), padding="same")
    s3_b2 = cov_bn(s3_b2, 96, 3, 3, strides=(2, 2), padding="valid")

    s3_b3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = layers.concatenate([s3_b1, s3_b2, s3_b3], axis=3, name="mix2")

    # 第4个并联开始
    s4_b1 = cov_bn(x, 192, 1, 1, strides=(1, 1), padding="same")

    s4_b2 = cov_bn(x, 128, 1, 1, strides=(1, 1), padding="same")
    s4_b2 = cov_bn(s4_b2, 128, 1, 7, strides=(1, 1), padding="same")
    s4_b2 = cov_bn(s4_b2, 192, 7, 1, strides=(1, 1), padding="same")

    s4_b3 = cov_bn(x, 128, 1, 1, strides=(1, 1), padding="same")
    s4_b3 = cov_bn(s4_b3, 128, 7, 1, strides=(1, 1), padding="same")
    s4_b3 = cov_bn(s4_b3, 128, 1, 7, strides=(1, 1), padding="same")
    s4_b3 = cov_bn(s4_b3, 128, 7, 1, strides=(1, 1), padding="same")
    s4_b3 = cov_bn(s4_b3, 192, 1, 7, strides=(1, 1), padding="same")

    s4_b4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    s4_b4 = cov_bn(s4_b4, 192, 1, 1, strides=(1, 1), padding="same")

    x = layers.concatenate([s4_b1, s4_b2, s4_b3, s4_b4], axis=3, name="mix3")

    # 第5个开始
    s5_b1 = cov_bn(x, 192, 1, 1, strides=(1, 1), padding="same")

    s5_b2 = cov_bn(x, 160, 1, 1, strides=(1, 1), padding="same")
    s5_b2 = cov_bn(s5_b2, 160, 1, 7, strides=(1, 1), padding="same")
    s5_b2 = cov_bn(s5_b2, 192, 7, 1, strides=(1, 1), padding="same")

    s5_b3 = cov_bn(x, 160, 1, 1, strides=(1, 1), padding="same")
    s5_b3 = cov_bn(s5_b3, 160, 7, 1, strides=(1, 1), padding="same")
    s5_b3 = cov_bn(s5_b3, 160, 1, 7, strides=(1, 1), padding="same")
    s5_b3 = cov_bn(s5_b3, 160, 7, 1, strides=(1, 1), padding="same")
    s5_b3 = cov_bn(s5_b3, 192, 1, 7, strides=(1, 1), padding="same")

    s5_b4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    s5_b4 = cov_bn(s5_b4, 192, 1, 1, strides=(1, 1), padding="same")

    x = layers.concatenate([s5_b1, s5_b2, s5_b3, s5_b4], axis=3, name="mix4")

    # 第六个
    s6_b1 = cov_bn(x, 192, 1, 1, strides=(1, 1), padding="same")

    s6_b2 = cov_bn(x, 160, 1, 1, strides=(1, 1), padding="same")
    s6_b2 = cov_bn(s6_b2, 160, 1, 7, strides=(1, 1), padding="same")
    s6_b2 = cov_bn(s6_b2, 192, 7, 1, strides=(1, 1), padding="same")

    s6_b3 = cov_bn(x, 160, 1, 1, strides=(1, 1), padding="same")
    s6_b3 = cov_bn(s6_b3, 160, 7, 1, strides=(1, 1), padding="same")
    s6_b3 = cov_bn(s6_b3, 160, 1, 7, strides=(1, 1), padding="same")
    s6_b3 = cov_bn(s6_b3, 160, 7, 1, strides=(1, 1), padding="same")
    s6_b3 = cov_bn(s6_b3, 192, 1, 7, strides=(1, 1), padding="same")

    s6_b4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    s6_b4 = cov_bn(s6_b4, 192, 1, 1, strides=(1, 1), padding="same")

    x = layers.concatenate([s6_b1, s6_b2, s6_b3, s6_b4], axis=3, name="mix5")

    # 第7个
    s7_b1 = cov_bn(x, 192, 1, 1, strides=(1, 1), padding="same")

    s7_b2 = cov_bn(x, 192, 1, 1, strides=(1, 1), padding="same")
    s7_b2 = cov_bn(s7_b2, 192, 1, 7, strides=(1, 1), padding="same")
    s7_b2 = cov_bn(s7_b2, 192, 7, 1, strides=(1, 1), padding="same")

    s7_b3 = cov_bn(x, 192, 1, 1, strides=(1, 1), padding="same")
    s7_b3 = cov_bn(s7_b3, 192, 7, 1, strides=(1, 1), padding="same")
    s7_b3 = cov_bn(s7_b3, 192, 1, 7, strides=(1, 1), padding="same")
    s7_b3 = cov_bn(s7_b3, 192, 7, 1, strides=(1, 1), padding="same")
    s7_b3 = cov_bn(s7_b3, 192, 1, 7, strides=(1, 1), padding="same")

    s7_b4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    s7_b4 = cov_bn(s7_b4, 192, 1, 1, strides=(1, 1), padding="same")

    x = layers.concatenate([s7_b1, s7_b2, s7_b3, s7_b4], axis=3, name="mix6")

    # 第8个
    s8_b1 = cov_bn(x, 192, 1, 1, strides=(1, 1), padding="same")
    s8_b1 = cov_bn(s8_b1, 320, 3, 3, strides=(2, 2), padding="valid")

    s8_b2 = cov_bn(x, 192, 1, 1, strides=(1, 1), padding="same")
    s8_b2 = cov_bn(s8_b2, 192, 1, 7, strides=(1, 1), padding="same")
    s8_b2 = cov_bn(s8_b2, 192, 7, 1, strides=(1, 1), padding="same")
    s8_b2 = cov_bn(s8_b2, 192, 3, 3, strides=(2, 2), padding="valid")

    s8_b3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = layers.concatenate([s8_b1, s8_b2, s8_b3], axis=3, name="mix7")

    # 第9个
    s9_b1 = cov_bn(x, 320, 1, 1, strides=(1, 1), padding="same")

    s9_b2 = cov_bn(x, 384, 1, 1, strides=(1, 1), padding="same")
    s9_b2_1 = cov_bn(s9_b2, 384, 1, 3, strides=(1, 1), padding="same")
    s9_b2_2 = cov_bn(s9_b2, 384, 3, 1, strides=(1, 1), padding="same")
    s9_b2 = layers.concatenate([s9_b2_1, s9_b2_2], axis=3)

    s9_b3 = cov_bn(x, 448, 1, 1, strides=(1, 1), padding="same")
    s9_b3 = cov_bn(s9_b3, 384, 3, 3, strides=(1, 1), padding="same")
    s9_b3_1 = cov_bn(s9_b3, 384, 1, 3, strides=(1, 1), padding="same")
    s9_b3_2 = cov_bn(s9_b3, 384, 3, 1, strides=(1, 1), padding="same")
    s9_b3 = layers.concatenate([s9_b3_1, s9_b3_2], axis=3)

    s9_b4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    s9_b4 = cov_bn(s9_b4, 192, 1, 1, strides=(1, 1), padding="same")

    x = layers.concatenate([s9_b1, s9_b2, s9_b3, s9_b4], axis=3, name="mix8")

    # 第10个
    s10_b1 = cov_bn(x, 320, 1, 1, strides=(1, 1), padding="same")

    s10_b2 = cov_bn(x, 384, 1, 1, strides=(1, 1), padding="same")
    s10_b2_1 = cov_bn(s10_b2, 384, 1, 3, strides=(1, 1), padding="same")
    s10_b2_2 = cov_bn(s10_b2, 384, 3, 1, strides=(1, 1), padding="same")
    s10_b2 = layers.concatenate([s10_b2_1, s10_b2_2], axis=3)

    s10_b3 = cov_bn(x, 448, 1, 1, strides=(1, 1), padding="same")
    s10_b3 = cov_bn(s10_b3, 384, 3, 3, strides=(1, 1), padding="same")
    s10_b3_1 = cov_bn(s10_b3, 384, 1, 3, strides=(1, 1), padding="same")
    s10_b3_2 = cov_bn(s10_b3, 384, 3, 1, strides=(1, 1), padding="same")
    s10_b3 = layers.concatenate([s10_b3_1, s10_b3_2], axis=3)

    s10_b4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    s10_b4 = cov_bn(s10_b4, 192, 1, 1, strides=(1, 1), padding="same")

    x = layers.concatenate([s10_b1, s10_b2, s10_b3, s10_b4], axis=3, name="mix9")

    x = GlobalAveragePooling2D()(x)

    x = Dense(classes, activation="softmax", name="predict")(x)

    inputs = image_input

    model = Model(inputs, x, name="inception_v3")

    return model


if __name__ == '__main__':
    model = inceptionV3Model()
    model.summary()
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    load_img = image.load_img("elephant.jpg", target_size=(299, 299))
    xarray = image.img_to_array(load_img)
    dims = np.expand_dims(xarray, axis=0)
    x = preprocess_input(dims)
    predict = model.predict(x)
    print("predict", decode_predictions(predict))
