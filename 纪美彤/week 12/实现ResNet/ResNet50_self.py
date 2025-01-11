# ResNet50有两个基本的块，分别名为Conv Block和Identity Block，其中Conv Block输入和输出的维度
# 是不一样的，所以不能连续串联，它的作用是改变网络的维度；Identity Block输入维度和输出维度相
# 同，可以串联，用于加深网络的。

import numpy as np
from keras import layers

from keras.layers import Input
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import Activation,BatchNormalization,Flatten
from keras.models import Model

from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
# 先实现Conv Block和Identity Block
# conv_block左分支经过三个conv2d，3个batchnorm和2个relu，右分支经过一个conv2d和一个batchnorm，左右分支输出尺寸保持一致，相加后共同经过一个relu后输出
def conv_block_self(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    # 命名方式
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 1*1卷积用于实现特征通道的升维和降维
    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    # padding = same, strid = 1,输出hw不变
    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

# conv_block左分支经过3个conv2d，3个batchnorm和2个relu，右分支直接输出输入张量，左右分支输出尺寸保持一致，相加后共同经过一个relu后输出
def identity_block_self(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

# 实现resnet50
def ResNet50_self(input_shape=[224,224,3],classes=1000):

    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(img_input)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block_self(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block_self(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block_self(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block_self(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block_self(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block_self(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block_self(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block_self(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block_self(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block_self(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block_self(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block_self(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block_self(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block_self(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block_self(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block_self(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')


    return model

if __name__ == '__main__':
    # 读取参数
    model = ResNet50_self()
    model.summary()
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
