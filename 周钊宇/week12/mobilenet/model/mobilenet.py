import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, DepthwiseConv2D,GlobalAveragePooling2D,Reshape,Dropout
from keras.models import Model

def ConvBlock(inputs, filters, kernel_size=(3,3), strides=(1,1)):

    x = Conv2D(filters= filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def DepthwiseConvBlock(inputs, filters, depth_mulipliter=1, strides=(1,1)):
    x = DepthwiseConv2D(kernel_size=(3,3), strides=strides, padding='same',
                        depth_multiplier=depth_mulipliter,use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1),
               padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def MobileNet(input_shape=(224,224,3), num_classes=1000, depth_multiplier=1, dropout = 0.001):

    input_tensor = Input(input_shape)

    x = ConvBlock(inputs= input_tensor, filters=32, strides=(2,2)) #112,112,32
    x = DepthwiseConvBlock(inputs=x, filters=64, depth_mulipliter=depth_multiplier)#112,112,64
    x = DepthwiseConvBlock(x, filters=128, depth_mulipliter=depth_multiplier, strides=(2,2)) #56,56,128
    x = DepthwiseConvBlock(x, filters=128, depth_mulipliter=depth_multiplier)#56,56,128
    x = DepthwiseConvBlock(x, filters=256, depth_mulipliter=depth_multiplier, strides=(2,2))#28,28,256

    x = DepthwiseConvBlock(x, depth_mulipliter=depth_multiplier,filters=256) #28,28,256
    x = DepthwiseConvBlock(x, filters=512, depth_mulipliter=depth_multiplier, strides=(2,2)) #14,14,512
    x = DepthwiseConvBlock(x, filters=512, depth_mulipliter=depth_multiplier)
    x = DepthwiseConvBlock(x, filters=512, depth_mulipliter=depth_multiplier)
    x = DepthwiseConvBlock(x, filters=512, depth_mulipliter=depth_multiplier)
    x = DepthwiseConvBlock(x, filters=512, depth_mulipliter=depth_multiplier)
    x = DepthwiseConvBlock(x, filters=512, depth_mulipliter=depth_multiplier)#14,14,512

    x = DepthwiseConvBlock(x, filters=1024, depth_mulipliter=depth_multiplier,strides=(2,2)) #7,7,1024
    x = DepthwiseConvBlock(x, filters=1024, depth_mulipliter=depth_multiplier) #7,7,1024

    x = GlobalAveragePooling2D()(x) #1024
    x = Reshape((1,1,1024))(x)#1,1 1024
    x = Dropout(dropout)(x)
    x = Conv2D(num_classes, kernel_size=(1,1), padding='same')(x) # 1,1 1000
    x = Activation('softmax')(x)
    x = Reshape((num_classes,))(x) #1000
    


    model = Model(inputs=input_tensor, outputs=x)
    model.load_weights("mobilenet_1_0_224_tf.h5")
    return model

# model= MobileNet()
# model.summary()