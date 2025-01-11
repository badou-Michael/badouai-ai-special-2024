import numpy as np

from keras import layers
from keras.layers import Input
from keras.layers import Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import BatchNormalization, Activation, Flatten

from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions

def Identity_Block(input, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input])
    x = Activation('relu')(x)
    return x

def Conv_Block(input, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    y = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input)
    y = BatchNormalization(name=bn_name_base + '1')(y)
    x = layers.add([x, y])
    x = Activation('relu')(x)
    return x

def ResNet_50(input_shape=None, classes=1000):
    if input_shape is None:
        input_shape = [224, 224, 3]
    inputs = Input(shape=input_shape)
    # add padding to make sure output and input dimension match
    x = ZeroPadding2D((3, 3))(inputs)

    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = Conv_Block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = Identity_Block(x, 3, [64, 64, 256], stage=2, block='b')
    x = Identity_Block(x, 3, [64, 64, 256], stage=2, block='c')

    x = Conv_Block(x, 3, [128, 128, 512], stage=3, block='a')
    x = Identity_Block(x, 3, [128, 128, 512], stage=3, block='b')
    x = Identity_Block(x, 3, [128, 128, 512], stage=3, block='c')
    x = Identity_Block(x, 3, [128, 128, 512], stage=3, block='d')

    x = Conv_Block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = Identity_Block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = Identity_Block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = Identity_Block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = Identity_Block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = Identity_Block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = Conv_Block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = Identity_Block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = Identity_Block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), strides=(7, 7), name='avg_pool')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(classes, activation='softmax', name='fc1')(x)

    model = Model(inputs=inputs, outputs=x, name='resnet50')

    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    return model

if __name__ == '__main__':
    model = ResNet_50()
    model.summary()
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    y = model.predict(x)
    print(decode_predictions(y))
