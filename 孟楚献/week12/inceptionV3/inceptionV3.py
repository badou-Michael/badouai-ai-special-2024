import numpy as np
from dask.dataframe.io.demo import names
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, concatenate, \
    GlobalAveragePooling2D, Dense
from keras.preprocessing import image
from tensorflow_core.python.keras.applications.densenet import decode_predictions


# 卷积，批量归一化，激活一条龙
def conv2d_bn_activate(x, filters, kernel_size, strides=(1, 1), padding="same", name=None):
    if name is not None:
        bn_name = name + "_bn"
        conv_name = name + "_conv"
    else:
        bn_name = None
        conv_name = None
    # BatchNormalization有bias的效果
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, name=conv_name, use_bias=False)(x)
    x = BatchNormalization(name=bn_name, scale=False)(x)
    x = Activation("relu", name=name)(x)
    return x

def InfeptionV3(input_shape=[299, 299, 3], classes=1000):
    img_input = Input(input_shape)

    x = conv2d_bn_activate(img_input, 32, (3, 3), strides=(2, 2), padding='valid')
    x = conv2d_bn_activate(x, 32, (3, 3))
    x = conv2d_bn_activate(x, 64, (3, 3))
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn_activate(x, 80, (1, 1), padding='valid')
    x = conv2d_bn_activate(x, 192, (3, 3), padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # ----------------
    #   Block1 35x35
    # ----------------
    # Block1 module1
    # 35 x 35 x 192 -> 35 x 35 x 256
    branch1x1 = conv2d_bn_activate(x, 64, (1, 1))

    branch5x5 = conv2d_bn_activate(x, 48, (1, 1))
    branch5x5 = conv2d_bn_activate(branch5x5, 64, (5, 5))

    branch3x3dbl = conv2d_bn_activate(x, 64, (1, 1))
    branch3x3dbl = conv2d_bn_activate(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv2d_bn_activate(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn_activate(branch_pool, 32, (1, 1))
    # 64 + 64 + 96 + 32 = 256
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name="mixed0")

    # Block1 module2
    # 35 x 35 x 256 -> 35 x 35 x 288
    branch1x1 = conv2d_bn_activate(x, 64, (1, 1))

    branch5x5 = conv2d_bn_activate(x, 48, (1, 1))
    branch5x5 = conv2d_bn_activate(branch5x5, 64, (5, 5))

    branch3x3dbl = conv2d_bn_activate(x, 64, (1, 1))
    branch3x3dbl = conv2d_bn_activate(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv2d_bn_activate(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn_activate(branch_pool, 64, (1, 1))

    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name="mixed1")

    # Block1 module3
    # 35 x 35 x 288 -> 35 x 35 x 288
    branch1x1 = conv2d_bn_activate(x, 64, (1, 1))

    branch5x5 = conv2d_bn_activate(x, 48, (1, 1))
    branch5x5 = conv2d_bn_activate(branch5x5, 64, (5, 5))

    branch3x3dbl = conv2d_bn_activate(x, 64, (1, 1))
    branch3x3dbl = conv2d_bn_activate(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv2d_bn_activate(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn_activate(branch_pool, 64, (1, 1))

    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name="mixed2")

    # ----------------
    #   Block2 17x17
    # ----------------
    # Block2 module1
    # 35 x 35 x 288 -> 17 x 17 x 768
    branch3x3 = conv2d_bn_activate(x, 384, (3, 3), (2, 2), padding='valid')

    branch3x3dbl = conv2d_bn_activate(x, 64, (1, 1))
    branch3x3dbl = conv2d_bn_activate(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv2d_bn_activate(branch3x3dbl, 96, (3, 3), (2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(x)

    x = concatenate([branch3x3, branch3x3dbl, branch_pool], axis=3, name="mixed3")

    # Block2 module2
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn_activate(x, 192, (1, 1))

    branch7x7 = conv2d_bn_activate(x, 128, (1, 1), name="128_1")
    branch7x7 = conv2d_bn_activate(branch7x7, 128, (1, 7), name="128_2")
    branch7x7 = conv2d_bn_activate(branch7x7, 192, (7, 1),  name="128_3")

    branch7x7dbl = conv2d_bn_activate(x, 128, (1, 1), name="128_4")
    branch7x7dbl = conv2d_bn_activate(branch7x7dbl, 128, (7, 1))
    branch7x7dbl = conv2d_bn_activate(branch7x7dbl, 128, (1, 7),  name="128_5")
    branch7x7dbl = conv2d_bn_activate(branch7x7dbl, 128, (7, 1))
    branch7x7dbl = conv2d_bn_activate(branch7x7dbl, 192, (1, 7))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv2d_bn_activate(branch_pool, 192, (1, 1))

    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name="mixed4")

    # Block2 module3 4
    # 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn_activate(x, 192, (1, 1))

        branch7x7 = conv2d_bn_activate(x, 160, (1, 1))
        branch7x7 = conv2d_bn_activate(branch7x7, 160, (1, 7))
        branch7x7 = conv2d_bn_activate(branch7x7, 192, (7, 1))

        branch7x7dbl = conv2d_bn_activate(x, 160, (1, 1))
        branch7x7dbl = conv2d_bn_activate(branch7x7dbl, 160, (7, 1))
        branch7x7dbl = conv2d_bn_activate(branch7x7dbl, 160, (1, 7))
        branch7x7dbl = conv2d_bn_activate(branch7x7dbl, 160, (7, 1))
        branch7x7dbl = conv2d_bn_activate(branch7x7dbl, 192, (1, 7))

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = conv2d_bn_activate(branch_pool, 192, (1, 1))

        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name="mixed"+str(5+i))

    # Block2 module5
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv2d_bn_activate(x, 192, (1, 1))

    branch7x7 = conv2d_bn_activate(x, 192, (1, 1))
    branch7x7 = conv2d_bn_activate(branch7x7, 192, (1, 7))
    branch7x7 = conv2d_bn_activate(branch7x7, 192, (7, 1))

    branch7x7dbl = conv2d_bn_activate(x, 192, (1, 1))
    branch7x7dbl = conv2d_bn_activate(branch7x7dbl, 192, (7, 1))
    branch7x7dbl = conv2d_bn_activate(branch7x7dbl, 192, (1, 7))
    branch7x7dbl = conv2d_bn_activate(branch7x7dbl, 192, (7, 1))
    branch7x7dbl = conv2d_bn_activate(branch7x7dbl, 192, (1, 7))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv2d_bn_activate(branch_pool, 192, (1, 1))

    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name="mixed7")

    # ----------------
    #   Block3 8x8
    # ----------------
    # Block3 module1
    # 17 x 17 x 768 -> 8 x 8 x 1280
    branch3x3 = conv2d_bn_activate(x, 192, (1, 1))
    branch3x3 = conv2d_bn_activate(branch3x3, 320, (3, 3), strides=(2, 2), padding="valid")

    branch7x7x3 = conv2d_bn_activate(x, 192, (1, 1))
    branch7x7x3 = conv2d_bn_activate(branch7x7x3, 192, (1, 7))
    branch7x7x3 = conv2d_bn_activate(branch7x7x3, 192, (7, 1))
    branch7x7x3 = conv2d_bn_activate(branch7x7x3, 192, (3, 3), strides=(2, 2), padding="valid")

    branch_pool = AveragePooling2D((3, 3), (2, 2), padding="valid")(x)
    # 320 + 192 + 768 = 1280
    x = concatenate([branch3x3, branch7x7x3, branch_pool], axis=3, name="mixed8")

    # Block3 module2 3
    # 8 x 8 x 1280 -> 8 x 8 x 2480 -> 8 x 8 x 2480
    for i in range(2):
        branch1x1 = conv2d_bn_activate(x, 320, (1, 1))

        branch3x3 = conv2d_bn_activate(x, 384, (1, 1))
        branch3x3_1 = conv2d_bn_activate(branch3x3, 384, (1, 3))
        branch3x3_2 = conv2d_bn_activate(branch3x3, 384, (3, 1))
        branch3x3 = concatenate([branch3x3_1, branch3x3_2], axis=3, name="mixed9_"+str(i))

        branch3x3dbl = conv2d_bn_activate(x, 448, (1, 1))
        branch3x3dbl = conv2d_bn_activate(branch3x3dbl, 384, (3, 3))
        branch3x3dbl_1 = conv2d_bn_activate(branch3x3dbl, 384, (1, 3))
        branch3x3dbl_2 = conv2d_bn_activate(branch3x3dbl, 384, (3, 1))
        branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = AveragePooling2D((3, 3), (1, 1), padding="same")(x)
        branch_pool = conv2d_bn_activate(branch_pool, 192, (1, 1))
        x = concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3, name="mixed"+str(9+i))

    # 全连接
    x = GlobalAveragePooling2D(name="avg_pool")(x)
    x = Dense(units=classes, activation="softmax", name="predictions")(x)

    inputs = img_input

    model = Model(inputs, x, name="inceptionV3")

    return model

def process_image(img):
    img /= 255.
    img -= 0.5
    img *= 2.
    return img

if __name__ == "__main__":
    model = InfeptionV3()
    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    print(x.shape)

    x = process_image(x)

    preds = model.predict(x)
    print("预测结果", decode_predictions(preds))


