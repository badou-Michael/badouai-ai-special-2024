from keras import layers
from keras.layers import Input
from keras.layers import Dense,Conv2D,BatchNormalization,Activation,ZeroPadding2D,MaxPool2D,AveragePooling2D,Flatten
from keras.models import Model

from keras.preprocessing import image

import numpy as np

from keras.applications.imagenet_utils import preprocess_input,decode_predictions

def identity_block(input_tensor,kernel_size,filters,stage,block):
    filters1,filters2,filters3 = filters

    conv_name_base = 'res'+str(stage)+block+'_branch'
    bn_name_base ='bn'+str(stage)+block+'_branch'

    # 1 输出通道数：filters1
    x = Conv2D(filters1,(1,1),name=conv_name_base+'2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    # 2 输出通道数：filters2
    # kernel_size:卷积核尺寸，可以设为1个int型数或者一个(int,int)型的元组。例如（2,3）是高2宽3卷积核
    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    # 3 输出通道数：filters3
    x = Conv2D(filters3, (1,1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 结果合并。
    x = layers.add([x,input_tensor])
    x = Activation('relu')(x)

    return x

def conv_block(input_tensor,kernel_size,filters,stage,block,strides=(2,2)):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 1
    x = Conv2D(filters1,(1,1),strides=strides,name=conv_name_base+'2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base+'2a')(x)
    x = Activation('relu')(x)

    # 2
    x = Conv2D(filters2,kernel_size,padding='same',name=conv_name_base+'2b')(x)
    x = BatchNormalization(name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    # 3
    x = Conv2D(filters3,(1,1),name=conv_name_base+'2c')(x)
    x = BatchNormalization(name=bn_name_base+'2c')(x)

    shortcut = Conv2D(filters3,(1,1),strides=strides,name=conv_name_base+'1')(input_tensor)
    shortcut_bn = BatchNormalization(name=bn_name_base+'1')(shortcut)
    # 合并
    x = layers.add([x,shortcut_bn])
    x = Activation('relu')(x)

    return x

def resnet50(input_shape=[224,224,3],classes=1000):
    # Input():初始化网络输入层的tensor
    inputs = Input(shape=input_shape)
    x = ZeroPadding2D((3,3))(inputs)

    # 1 output_size=112*112
    x = Conv2D(64,(7,7),strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    # output_size=56*56
    x = MaxPool2D((3,3),strides=(2,2))(x)

    # output_size=56*56
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # output_size=28*28
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # output_size=14*14
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # output_size=7*7
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # output_size=1*1
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(classes,activation='softmax', name='fc1000')(x)

    model = Model(inputs,x,name='resnet50')
    return model


if __name__=='__main__':
    model = resnet50()
    model.load_weights('E:/YAN/HelloWorld/cv/【12】图像识别/代码/resnet50_tf/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    model.summary()  # 会执行过程中把模型结构打印出来
    img = image.load_img('E:/YAN/HelloWorld/cv/【12】图像识别/代码/resnet50_tf/elephant.jpg',target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    x = preprocess_input(img) #归一化
    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:',decode_predictions(preds))# decode_predictions=vgg6_utils.print_prob
