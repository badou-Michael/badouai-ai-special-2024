
from keras import layers
from keras.layers import Activation,ZeroPadding2D,Input,Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,AveragePooling2D
from keras.models import Model
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input,decode_predictions

"""
 conv_block 
 input_tensor 张量x
 kernel_size 卷积核大小
 filters 表示3个卷积层的滤波器数量
 strides 步长
"""

def conv_block(input_tensor,kernel_size,filters,strides=(2,2)):

    #将filters的值分别给了filters1,filters2,filters3. 比如filters=[64, 64, 256] ，filters1=64，filters2=64，filters3=256
    filters1,filters2, filters3 = filters
    # str(stage) 将stage的值转化成string,实际layer不需要手动命名，手动命名还要考虑唯一性


    #(1,1) 用于降维，[64, 64, 256]减少通道数为64
    x = Conv2D(filters1,(1,1),strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2,kernel_size,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # (1,1) 用于升维，[64, 64, 256]提高通道数为256
    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters3,(1,1),strides=strides)(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    #残差连接
    x = layers.add([x,shortcut])
    x = Activation('relu')(x)
    return x

def identity_block(input_tensor,kernel_size,filters):
    filters1, filters2, filters3 = filters

    x =Conv2D(filters1,(1,1))(input_tensor)
    x = BatchNormalization()(x)
    x=Activation('relu')(x)

    x=Conv2D(filters2,kernel_size,padding='same')(x)
    x=BatchNormalization()(x)
    x=Activation('relu')(x)

    x=Conv2D(filters3,(1,1))(x)
    x=BatchNormalization()(x)

    x=layers.add([x,input_tensor])
    x=Activation('relu')(x)
    return x


#构建ResNet50模型，input_shape 输入图像大小， out_put_shape 分类数量
def ResNet50(input_shape=(224,224,3), out_put_shape=1000):
    #input->Zeropad->Conv2d、BatchNorm、Relu、MaxPool
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3,3))(img_input)
    # 初始卷积层7*7，步长2，输出64个通道，（x）将输入张量x传递给该层，完成卷积计算
    x = Conv2D(64,(7,7),strides=(2,2),name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3),strides=(2,2))(x)

    x = conv_block(x,3,[64,64,256],strides=(1,1))
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])

    x = conv_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])

    x = conv_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])

    x = conv_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(out_put_shape, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')

    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model

if __name__ == '__main__':
    model = ResNet50()
    model.summary()
    # img_path = 'elephant.jpg'
    img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))


