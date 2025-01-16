from __future__ import print_function
from __future__ import absolute_import
import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Activation,Dense,Input,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image


def conv_block(input_tensor, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, name=conv_name)(input_tensor)
    x = BatchNormalization(name=bn_name)(x)
    x = Activation('relu')(x)
    return x


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def InceptionV3(input_shape=[299, 299, 3], num_classes=1000):
    img_input = Input(shape=input_shape)
    x = conv_block(img_input, 32, (3, 3), (2, 2), 'valid')
    x = conv_block(x, 32, (3, 3), padding='valid')
    x = conv_block(x, 64, (3, 3))
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = conv_block(x, 80, (1, 1), padding='valid')
    x = conv_block(x, 192, (3, 3), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Block1 part1 35 x 35 x 192 -> 35 x 35 x 256
    branch1x1 = conv_block(x, 64, (1, 1))

    branch5x5 = conv_block(x, 48, (1, 1))
    branch5x5 = conv_block(branch5x5, 64, (5, 5))

    branch3x3dbl = conv_block(x, 64, (1, 1))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv_block(branch_pool, 32, (1, 1))

    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed0')

    # Block 1 part 2: 5 x 35 x 256 -> 35 x 35 x 288
    branch1x1 = conv_block(x, 64, (1, 1))

    branch5x5 = conv_block(x, 48, (1, 1))
    branch5x5 = conv_block(branch5x5, 64, (5, 5))

    branch3x3dbl = conv_block(x, 64, (1, 1))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv_block(branch_pool, 64, (1, 1))
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed1'
    )

    # Block 1 part 3 35 x 35 x 288 -> 35 x 35 x 288
    branch1x1 = conv_block(x, 64, (1, 1))

    branch5x5 = conv_block(x, 48, (1, 1))
    branch5x5 = conv_block(branch5x5, 64, (5, 5))

    branch3x3dbl = conv_block(x, 64, (1, 1))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv_block(branch_pool, 64, (1, 1))

    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed2'
    )

    # Block 2 part 1 35 x 35 x 288 -> 17 x 17 x 768
    branch3x3 = conv_block(x, 384, (3, 3), strides=(2, 2), padding='valid')
    branch3x3dbl = conv_block(x, 64, (1, 1))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3), strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed3'
    )

    # Block 2 part 2
    branch1x1 = conv_block(x, 192, (1, 1))

    branch7x7 = conv_block(x, 128, (1, 1))
    branch7x7 = conv_block(branch7x7, 128, (1, 7))
    branch7x7 = conv_block(branch7x7, 192, (7, 1))

    branch7x7dbl = conv_block(x, 128, (1, 1))
    branch7x7dbl = conv_block(branch7x7dbl, 128, (7, 1))
    branch7x7dbl = conv_block(branch7x7dbl, 128, (1, 7))
    branch7x7dbl = conv_block(branch7x7dbl, 128, (7, 1))
    branch7x7dbl = conv_block(branch7x7dbl, 192, (1, 7))

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv_block(branch_pool, 192, (1, 1))
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed4'
    )

    # Block2 part3 and part4 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv_block(x, 192, (1, 1))

        branch7x7 = conv_block(x, 160, (1, 1))
        branch7x7 = conv_block(branch7x7, 160, (1, 7))
        branch7x7 = conv_block(branch7x7, 192, (7, 1))

        branch7x7dbl = conv_block(x, 160, (1, 1))
        branch7x7dbl = conv_block(branch7x7dbl, 160, (7, 1))
        branch7x7dbl = conv_block(branch7x7dbl, 160, (1, 7))
        branch7x7dbl = conv_block(branch7x7dbl, 160, (7, 1))
        branch7x7dbl = conv_block(branch7x7dbl, 192, (1, 7))

        branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
        branch_pool = conv_block(branch_pool, 192, (1, 1))
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed' + str(5 + i)
        )

    # Block 2 part 5: 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv_block(x, 192, (1, 1))

    branch7x7 = conv_block(x, 192, (1, 1))
    branch7x7 = conv_block(branch7x7, 192, (1, 7))
    branch7x7 = conv_block(branch7x7, 192, (7, 1))

    branch7x7dbl = conv_block(x, 192, (1, 1))
    branch7x7dbl = conv_block(branch7x7dbl, 192, (7, 1))
    branch7x7dbl = conv_block(branch7x7dbl, 192, (1, 7))
    branch7x7dbl = conv_block(branch7x7dbl, 192, (7, 1))
    branch7x7dbl = conv_block(branch7x7dbl, 192, (1, 7))

    branch_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)
    branch_pool = conv_block(branch_pool, 192, (1, 1))
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed7'
    )

    # Block 3 part 1 17 x 17 x 768 -> 8 x 8 x 1280
    branch3x3 = conv_block(x, 192, (1, 1))
    branch3x3 = conv_block(branch3x3, 320, (3, 3), strides=(2, 2), padding='valid')

    branch7x7x3 = conv_block(x, 192, (1, 1))
    branch7x7x3 = conv_block(branch7x7x3, 192, (1, 7))
    branch7x7x3 = conv_block(branch7x7x3, 192, (7, 1))
    branch7x7x3 = conv_block(branch7x7x3, 192, (3, 3), strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=3,
        name='mixed8'
    )

    # Block 3 part 2, 3 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv_block(x, 320, (1, 1))

        branch3x3 = conv_block(x, 384, (1, 1))
        branch3x3_1 = conv_block(branch3x3, 384, (1, 3))
        branch3x3_2 = conv_block(branch3x3, 384, (3, 1))
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

        branch3x3dbl = conv_block(x, 448, (1, 1))
        branch3x3dbl = conv_block(branch3x3dbl, 384, (3, 3))
        branch3x3dbl_1 = conv_block(branch3x3dbl, 384, (1, 3))
        branch3x3dbl_2 = conv_block(branch3x3dbl, 384, (3, 1))
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv_block(branch_pool, 192, (1, 1))
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed' + str(9 + i))

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x)

    input = img_input
    model = Model(input, x, name='inception_v3')
    return model


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
