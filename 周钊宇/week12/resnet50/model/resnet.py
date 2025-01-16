import keras
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, MaxPooling2D, Activation, Dense,Flatten,AveragePooling2D
from keras.models import Model

def ConvBlock(input_tensor, filters, strides):
    filter1, filter2, filter3 = filters
    x = Conv2D(filters=filter1,
               kernel_size=(1,1),
               strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filter2,
               kernel_size=(3,3),
               strides=(1,1),
               padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filter3,
               kernel_size=(1,1),
               strides=(1,1))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters=filter3,
                      kernel_size=(1,1),
                      strides=strides)(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    output = keras.layers.add([x,shortcut])
    output = Activation('relu')(output)
    return output

def IdentityBlock(input_tensor, filters):
    filter1, filter2, filter3 = filters
    x = Conv2D(filters=filter1, kernel_size=(1,1), strides=(1,1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filter2, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filter3, kernel_size=(1,1), strides=(1,1))(x)
    x = BatchNormalization()(x)

    output = keras.layers.add([input_tensor, x])
    output = Activation('relu')(output)

    return output

def resnet50(input_tensor=(224, 224, 3), classes=1000):
    inputs = Input(input_tensor)
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2D(filters=64,
               kernel_size=(7, 7),
               strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3),
                     strides=(2, 2),
                     padding='same')(x)
    #stage1
    x = ConvBlock(input_tensor=x, filters=[64,64,256], strides=(1,1))
    x = IdentityBlock(input_tensor=x, filters=[64,64,256])
    x = IdentityBlock(input_tensor=x, filters=[64,64,256])

    #stage2
    x = ConvBlock(input_tensor=x, filters=[128,128,512], strides=(2,2))
    x = IdentityBlock(input_tensor=x, filters=[128,128,512])
    x = IdentityBlock(input_tensor=x, filters=[128,128,512])
    x = IdentityBlock(input_tensor=x, filters=[128,128,512])

    #stage3
    x = ConvBlock(input_tensor=x, filters=[256,256,1024], strides=(2,2))
    x = IdentityBlock(input_tensor=x, filters=[256,256,1024])
    x = IdentityBlock(input_tensor=x, filters=[256,256,1024])
    x = IdentityBlock(input_tensor=x, filters=[256,256,1024])
    x = IdentityBlock(input_tensor=x, filters=[256,256,1024])
    x = IdentityBlock(input_tensor=x, filters=[256,256,1024])


    #stage4
    x = ConvBlock(input_tensor=x, filters=[512, 512, 2048], strides=(2,2))
    x = IdentityBlock(input_tensor=x, filters=[512, 512, 2048])
    x = IdentityBlock(input_tensor=x, filters=[512, 512, 2048])

    x = AveragePooling2D((7,7))(x)
    x = Flatten()(x)

    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    return model


# model = resnet50()
# model.summary()
