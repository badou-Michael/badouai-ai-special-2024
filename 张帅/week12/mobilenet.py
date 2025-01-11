import warnings
import numpy as np

from keras.preprocessing import image

from keras.models import Model
from keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D, Dense
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input

def conv2d_bn(input_x, filters, strides=(1, 1), padding='same', name=None):
    conv_name = name if name is None else f'{name}_conv'
    bn_name = name if name is None else f'{name}_bn'

    x = Conv2D(filters, (3, 3), strides=strides, padding=padding, use_bias=False, name=conv_name)(input_x)
    x = BatchNormalization(name=bn_name)(x)
    x = Activation(relu6, name=name)(x)
    return x
def conv_dw(input_x, filters, strides=(1, 1),name=None):
    conv_name = name if name is None else f'{name}_conv_dw'
    bn_name = name if name is None else f'{name}_bn_dw'
    act_name = name if name is None else f'{name}_act_dw'

    x = DepthwiseConv2D((3, 3),strides=strides, padding='same',depth_multiplier=1,use_bias=False, name=f'{conv_name}_1')(input_x)
    x = BatchNormalization(name=f'{bn_name}_1')(x)
    x = Activation(relu6, name=f'{act_name}_1')(x)

    x = Conv2D(filters, (1, 1), strides=(1,1), padding='same', use_bias=False, name=f'{conv_name}_2')(x)
    x = BatchNormalization(name=f'{bn_name}_2')(x)
    x = Activation(relu6, name=f'{act_name}_2')(x)
    return x

def relu6(x):
    return K.relu(x, max_value=6)

def mobile_net(input_shape=[224, 224, 3]):
    img_input = Input(shape=input_shape)
    x = conv2d_bn(img_input, 32, strides=(2,2), name='n1')
    x = conv_dw(x, 64, name='n2')
    x = conv_dw(x, 128, strides=(2,2), name='n3')
    x = conv_dw(x, 128,name='n4')
    x = conv_dw(x, 256, strides=(2,2),name='n5')
    x = conv_dw(x, 256,name='n6')
    x = conv_dw(x, 512, strides=(2,2),name='n7')
    for i in range(8, 13):
        x = conv_dw(x, 512, name='n%s'%i)
    x = conv_dw(x, 1024, strides=(2,2),name='n13')
    x = conv_dw(x, 1024, name='n14')

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,1024), name='reshape_1')(x)
    x = Dropout(1e-3, name='dropout_1')(x)
    x = Conv2D(1000, (1,1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='acc_softmax')(x)
    x = Reshape((1000,), name='reshape_2')(x)

    model = Model(img_input, x, name='mobilenet')
    model.load_weights('./mobilenet_1_0_224_tf.h5')

    return model


if __name__ == '__main__':
    model = mobile_net()

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds,1))  # 只显示top1





 108 changes: 108 additions & 0 deletions108  
