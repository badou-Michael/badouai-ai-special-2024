#-*- coding:utf-8 -*-
import numpy as np
from keras import layers, models
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions

def conv_block(x, filters, kernel_size, strides=(1, 1), padding='same', activation='relu'):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x

# Inception module implementation
def inception_module(x, filters):
    """
    Inception module that applies multiple branches of convolutions and pooling.
    """
    # Filters for each branch
    f1, f3r, f3, f5r, f5, fp = filters

    # 1x1 Convolution branch
    branch1x1 = conv_block(x, f1, (1, 1))

    # 1x1 Convolution followed by 3x3 Convolution branch
    branch3x3 = conv_block(x, f3r, (1, 1))
    branch3x3 = conv_block(branch3x3, f3, (3, 3))

    # 1x1 Convolution followed by 5x5 Convolution branch
    branch5x5 = conv_block(x, f5r, (1, 1))
    branch5x5 = conv_block(branch5x5, f5, (5, 5))

    # 3x3 Max Pooling followed by 1x1 Convolution branch
    branch_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_block(branch_pool, fp, (1, 1))

    # Concatenate all branches
    x = layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1)
    return x

def inceptionV2(input_shape=(299, 299, 3), num_classes=1000):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution layers (Conv -> Conv -> Conv padded)
    x = conv_block(inputs, 32, (3, 3), strides=(2, 2), padding='valid')  # Output: 149x149x32
    x = conv_block(x, 32, (3, 3), strides=(1, 1), padding='valid')       # Output: 147x147x32
    x = conv_block(x, 64, (3, 3), strides=(1, 1))                       # Output: 147x147x64

    # MaxPooling layer
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)  # Output: 73x73x64

    # Convolution layers after pooling
    x = conv_block(x, 80, (3, 3), strides=(1, 1), padding='valid')      # Output: 71x71x80
    x = conv_block(x, 192, (3, 3), strides=(2, 2), padding='valid')     # Output: 35x35x192

    # First set of Inception modules (3x Inception)
    x = inception_module(x, [64, 64, 64, 64, 96, 32])                  # Output: 35x35x288
    x = inception_module(x, [64, 64, 96, 64, 96, 64])
    x = inception_module(x, [64, 64, 96, 64, 96, 64])

    # Second set of Inception modules (5x Inception)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x) # Output: 17x17x288
    x = inception_module(x, [192, 128, 192, 128, 192, 128])            # Output: 17x17x768
    x = inception_module(x, [192, 128, 192, 128, 192, 128])
    x = inception_module(x, [192, 128, 192, 128, 192, 128])
    x = inception_module(x, [192, 128, 192, 128, 192, 128])
    x = inception_module(x, [192, 128, 192, 128, 192, 128])

    # Third set of Inception modules (2x Inception)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x) # Output: 8x8x768
    x = inception_module(x, [320, 160, 320, 160, 320, 160])            # Output: 8x8x1280
    x = inception_module(x, [320, 160, 320, 160, 320, 160])

    # Final pooling and classification
    x = layers.GlobalAveragePooling2D()(x)                             # Output: 1x1x1280
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)       # Output: 1x1x1000

    # Create model
    model = models.Model(inputs, outputs)
    return model

def do_preprocess_input(x): # (-1,1)
    x /= 255.
    x = -0.5
    x *= 2.
    return x

if __name__ == '__main__':
    model = inceptionV2(input_shape=(299, 299, 3), num_classes=1000)
    # model.load_weights('./inception_v3_weights_tf_dim_ordering_tf_kernels.h5')

    img_path = './elephant.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = do_preprocess_input(x)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))