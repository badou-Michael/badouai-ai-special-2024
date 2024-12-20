import numpy as np
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, Activation, DepthwiseConv2D, GlobalAveragePooling2D, Dropout, \
    Reshape, add, ZeroPadding2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from keras.preprocessing import image
from keras_applications.imagenet_utils import decode_predictions
from keras import backend as K
from keras import layers

############################## Mobilenet ##############################
# 基础conv块设置
def conv_block(inputs,
               filters,
               kernel,
               padding='same',
               strides=(1,1),
               activation='relu',
               name=None,
               block=None):
    if name is not None:
        bn_name = name + '_bn' + block
        conv_name = name + '_conv' + block
        relu_name = name + '_relu' + block
    else:
        bn_name = None
        conv_name = None
        relu_name = None
    x = Conv2D(filters,kernel,
               padding=padding,
               use_bias=False,
               strides=strides,
               name=conv_name)(inputs)
    x = BatchNormalization(name=bn_name)(x)
    x = Activation(activation,name=relu_name)(x)
    return x

def depthwise_conv_block(inputs,
               pointwise_conv_filters,
               depth_multiplier=1,
               padding='same',
               strides=(1,1),
               name=None,
               activation=None,
               block_id=1):
    if name is not None:
        bn_name = name + 'bn'
        conv_name = name + 'conv'
        relu_name = name + 'relu'
    else:
        bn_name = None
        conv_name = None
        relu_name = None
    x = DepthwiseConv2D((3,3),
                        padding=padding,
                        depth_multiplier = depth_multiplier,
                        strides = strides,
                        use_bias=False,
                        name = conv_name + '_dw_%d' % block_id
                        )(inputs)
    x = BatchNormalization(name = bn_name + '_dw_%d' % block_id)(x)
    x = Activation(activation, name=relu_name + '_dw_%d' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding=padding,
               use_bias=False,
               strides=strides,
               name=conv_name + '_pw_%d' % block_id)(x)
    x = BatchNormalization(name=bn_name + '_pw_%d' % block_id)(x)
    x = Activation(activation, name=relu_name + '_pw_%d' % block_id)(x)
    return x

def relu6(x):
    return K.relu(x, max_value=6)

# Mobilenet模型定义
def MobileNet(input_shape=[224,224,3],
                depth_multiplier=1,
                dropout=1e-3,
                classes=1000
                ):
    img_input = Input(shape=input_shape)
    # 224,224,3 -> 112,112,32
    x = conv_block(img_input,32,(3,3),
                   strides = (2,2),# 下采样
                   activation=relu6,
                   name='mob')
    # 112,112,32 -> 112,112,64
    x = depthwise_conv_block(x, 64, depth_multiplier,
                             strides=(2, 2),
                             name='mob',
                             activation=relu6,
                             block_id=1)
    # 112,112,64 -> 56,56,128
    x = depthwise_conv_block(x, 128, depth_multiplier,
                             name='mob',
                             activation=relu6,
                             block_id=2)
    # 56,56,128 -> 56,56,128
    x = depthwise_conv_block(x, 128, depth_multiplier,
                             strides=(2, 2),
                             name='mob',
                             activation=relu6,
                             block_id=3)
    # 56,56,128 -> 28,28,256
    x = depthwise_conv_block(x, 256, depth_multiplier,
                             strides=(2, 2),
                             name='mob',
                             activation=relu6,
                             block_id=4)
    # 28,28,256 -> 28,28,256
    x = depthwise_conv_block(x, 256, depth_multiplier,
                             name='mob',
                             activation=relu6,
                             block_id=5)
    # 28,28,256 -> 14,14,512
    x = depthwise_conv_block(x, 512, depth_multiplier,
                             strides=(2, 2),
                             name='mob',
                             activation=relu6,
                             block_id=6)
    # 14,14,512 -> 14,14,512 * 5
    x = depthwise_conv_block(x, 512, depth_multiplier, activation=relu6, name='mob', block_id=7)
    x = depthwise_conv_block(x, 512, depth_multiplier, activation=relu6, name='mob', block_id=8)
    x = depthwise_conv_block(x, 512, depth_multiplier, activation=relu6, name='mob', block_id=9)
    x = depthwise_conv_block(x, 512, depth_multiplier, activation=relu6, name='mob', block_id=10)
    x = depthwise_conv_block(x, 512, depth_multiplier, activation=relu6, name='mob', block_id=11)

    # 14,14,512 -> 7,7,1024
    x = depthwise_conv_block(x, 1024, depth_multiplier,
                             strides=(2, 2),
                             name='mob',
                             activation=relu6,
                             block_id=12)
    x = depthwise_conv_block(x, 1024, depth_multiplier,
                             name='mob',
                             activation=relu6,
                             block_id=13)

    # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), name='preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    inputs = img_input

    model = Model(inputs, x, name='mobilenet_1_0_224_tf')
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)

    return model

############################## ResNet50 ##############################
# Conv Block
def _conv_block(inputs,
               filters,
               kernel,
               strides = (2,2),
               name='name_base'
               ):
    filters1, filters2, filters3 = filters
    # 图左
    x = conv_block(inputs, filters1, (1, 1),
                   strides=strides,
                   name=name,
                   block='2a'
                   )
    x = conv_block(inputs, filters2, kernel,
                   strides=strides,
                   name=name,
                   block='2b')
    x = Conv2D(filters3, (1, 1), name='name_base_conv' + '2c')(x)
    x = BatchNormalization(name='name_base_bn' + '2c')(x)
    #图右
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name='name_base_conv' + '1')(inputs)
    shortcut = BatchNormalization(name='name_base_bn' + '1')(shortcut)
    # +
    x = add([x,shortcut])
    x = Activation('relu')

def _identity_block(inputs,
               filters,
               kernel,
               strides = (2,2),
               name='name_base',
               ):
    filters1, filters2, filters3 = filters
    # 图左
    x = conv_block(inputs, filters1, (1, 1),
                   strides=strides,
                   name=name,
                   block='2a'
                   )
    x = conv_block(inputs, filters2, kernel,
                   strides=strides,
                   name=name,
                   block='2b')
    x = Conv2D(filters3, (1, 1), name='name_base_conv' + '2c')(x)
    x = BatchNormalization(name='name_base_bn' + '2c')(x)
    # +
    x = add([x,inputs])
    x = Activation('relu')

# ResNet50 模型定义
def ResNet50(input_shape=[224,224,3],classes=1000):
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3,3))(img_input)

    x = conv_block(img_input,64,(7,7),strides=(2,2),name='ResNet50')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = _conv_block(x, 3, [64, 64, 256], name='name_base_2a', strides=(1, 1))
    x = _identity_block(x, 3, [64, 64, 256], name='name_base_2b')
    x = _identity_block(x, 3, [64, 64, 256], name='name_base_2c')

    x = _conv_block(x, 3, [128, 128, 512], name='name_base_3a')
    x = _identity_block(x, 3, [128, 128, 512], name='name_base_3b')
    x = _identity_block(x, 3, [128, 128, 512], name='name_base_3c')
    x = _identity_block(x, 3, [128, 128, 512], name='name_base_3d')

    x = conv_block(x, 3, [256, 256, 1024], name='name_base_4a')
    x = _identity_block(x, 3, [256, 256, 1024], name='name_base_4b')
    x = _identity_block(x, 3, [256, 256, 1024], name='name_base_4c')
    x = _identity_block(x, 3, [256, 256, 1024], name='name_base_4d')
    x = _identity_block(x, 3, [256, 256, 1024], name='name_base_4e')
    x = _identity_block(x, 3, [256, 256, 1024], name='name_base_4f')

    x = _conv_block(x, 3, [512, 512, 2048], name='name_base_5a')
    x = _identity_block(x, 3, [512, 512, 2048], name='name_base_5b')
    x = _identity_block(x, 3, [512, 512, 2048], name='name_base_5c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')

    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model

############################## InceptionV3 ##############################
def InceptionV3(input_shape=[299, 299, 3],
                classes=1000):
    img_input = Input(shape=input_shape)

    x = conv_block(img_input, 32, (3, 3) , strides=(2, 2), padding='valid',activation='relu')
    x = conv_block(x, 32, (3, 3), padding='valid')
    x = conv_block(x, 64, (3, 3))
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 80, (1, 1), padding='valid')
    x = conv_block(x, 192, (3, 3), padding='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # --------------------------------#
    #   Block1 35x35
    # --------------------------------#
    # Block1 part1
    # 35 x 35 x 192 -> 35 x 35 x 256
    branch1x1 = conv_block(x, 64, (1, 1))

    branch5x5 = conv_block(x, 48, (1, 1))
    branch5x5 = conv_block(branch5x5, 64, (5, 5))

    branch3x3dbl = conv_block(x, 64, (1, 1))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_block(branch_pool, 32, (1, 1))

    # 64+64+96+32 = 256
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed0')

    # Block1 part2
    # 35 x 35 x 256 -> 35 x 35 x 288
    branch1x1 = conv_block(x, 64, (1, 1))

    branch5x5 = conv_block(x, 48, (1, 1))
    branch5x5 = conv_block(branch5x5, 64, (5, 5))

    branch3x3dbl = conv_block(x, 64, (1, 1))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_block(branch_pool, 64, (1, 1))

    # 64+64+96+64 = 288
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed1')

    # Block1 part3
    # 35 x 35 x 288 -> 35 x 35 x 288
    branch1x1 = conv_block(x, 64, (1, 1))

    branch5x5 = conv_block(x, 48, (1, 1))
    branch5x5 = conv_block(branch5x5, 64, (5, 5))

    branch3x3dbl = conv_block(x, 64, (1, 1))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_block(branch_pool, 64, (1, 1))

    # 64+64+96+64 = 288
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=3,
        name='mixed2')

    # --------------------------------#
    #   Block2 17x17
    # --------------------------------#
    # Block2 part1
    # 35 x 35 x 288 -> 17 x 17 x 768
    branch3x3 = conv_block(x, 384, (3, 3), strides=(2, 2), padding='valid')

    branch3x3dbl = conv_block(x, 64, (1, 1))
    branch3x3dbl = conv_block(branch3x3dbl, 96, (3, 3))
    branch3x3dbl = conv_block(
        branch3x3dbl, 96, (3, 3), strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')

    # Block2 part2
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv_block(x, 192, (1, 1))

    branch7x7 = conv_block(x, 128, (1, 1))
    branch7x7 = conv_block(branch7x7, 128, (1, 7))
    branch7x7 = conv_block(branch7x7, 192, (7, 1))

    branch7x7dbl = conv_block(x, 128, (1, 1))
    branch7x7dbl = conv_block(branch7x7dbl, 128, (7, 1))
    branch7x7dbl = conv_block(branch7x7dbl, 128, (1, 7))
    branch7x7dbl = conv_block(branch7x7dbl, 128, (7, 1))
    branch7x7dbl = conv_block(branch7x7dbl, 192, (1, 7))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_block(branch_pool, 192, (1, 1))
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed4')

    # Block2 part3 and part4
    # 17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768
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

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv_block(branch_pool, 192, (1, 1))
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed' + str(5 + i))

    # Block2 part5
    # 17 x 17 x 768 -> 17 x 17 x 768
    branch1x1 = conv_block(x, 192, (1, 1))

    branch7x7 = conv_block(x, 192, (1, 1))
    branch7x7 = conv_block(branch7x7, 192, (1, 7))
    branch7x7 = conv_block(branch7x7, 192, (7, 1))

    branch7x7dbl = conv_block(x, 192, (1, 1))
    branch7x7dbl = conv_block(branch7x7dbl, 192, (7, 1))
    branch7x7dbl = conv_block(branch7x7dbl, 192, (1, 7))
    branch7x7dbl = conv_block(branch7x7dbl, 192, (7, 1))
    branch7x7dbl = conv_block(branch7x7dbl, 192, (1, 7))

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv_block(branch_pool, 192, (1, 1))
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=3,
        name='mixed7')

    # --------------------------------#
    #   Block3 8x8
    # --------------------------------#
    # Block3 part1
    # 17 x 17 x 768 -> 8 x 8 x 1280
    branch3x3 = conv_block(x, 192, (1, 1))
    branch3x3 = conv_block(branch3x3, 320, (3, 3),
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv_block(x, 192, (1, 1))
    branch7x7x3 = conv_block(branch7x7x3, 192, (1, 7))
    branch7x7x3 = conv_block(branch7x7x3, 192, (7, 1))
    branch7x7x3 = conv_block(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

    # Block3 part2 part3
    # 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv_block(x, 320, (1, 1))

        branch3x3 = conv_block(x, 384, (1, 1))
        branch3x3_1 = conv_block(branch3x3, 384, (1, 3))
        branch3x3_2 = conv_block(branch3x3, 384, (3, 1))
        branch3x3 = conv_block.concatenate(
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
    # 全连接
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)

    inputs = img_input

    model = Model(inputs, x, name='inception_v3')

    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")

    return model

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__':
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))

    model1 = MobileNet(input_shape=(224, 224, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    preds = model1.predict(x)
    print('MobileNet-Predicted:', decode_predictions(preds,1))  # 只显示top1

    model2 = ResNet50()
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    preds = model2.predict(x)
    print('ResNet50-Predicted:', decode_predictions(preds))
    model = InceptionV3()

    model3 = InceptionV3()
    img1 = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img1)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    preds = model3.predict(x)
    print('ResNet50Predicted:', decode_predictions(preds))
