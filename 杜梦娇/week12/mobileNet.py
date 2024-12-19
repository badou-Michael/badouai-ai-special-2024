import numpy as np
from keras import layers
from keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Activation, Dense, AveragePooling2D
from keras.layers import MaxPooling2D, ZeroPadding2D, Input, Flatten, Reshape, Dropout
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.models import Model
from keras import backend as K
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# DepthwiseConv2D深度可分离卷积:这种卷积操作首先独立地对输入的每个通道应用普通卷积，
# 然后使用逐点卷积（pointwise convolution）将深度卷积的结果组合起来，以产生最终的输出

def relu6(x):
    return K.relu(x, max_value=6)

def conv_bn(inputs, filters, kernel=(3, 3), strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(filters, kernel_size=kernel, name=conv_name, strides=strides, padding=padding, use_bias=False)(inputs)
    x = BatchNormalization(name=bn_name)(x)
    x = Activation(relu6, name=name)(x)
    return x

def depthwiseConvBlock(inputs,filters, kernel_size=(3, 3), depth_multiplier=1, strides=(1, 1), block=1):
    #inputs: 输入张量。
    #filters: 逐点卷积层的过滤器数量
    #kernel_size: 卷积核大小
    #depth_multiplier: 深度乘数，用于控制深度卷积的输出通道数
    #strides: 深度卷积层的步长
    #block: 用于命名层的块标识符

    # 构建DepthwiseConv
    x = DepthwiseConv2D(kernel_size=kernel_size, padding='same', depth_multiplier=depth_multiplier, strides=strides
                        , use_bias=False, name='conv_dw_%d' % block)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block)(x)
    x = Activation(relu6, name='conv_dw_%d_bn_relu' % block)(x)
    # 构建PointwiseConv
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%d' % block)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block)(x)
    x = Activation(relu6, name='conv_pw_%d_bn_relu' % block)(x)
    return x

# 定义inceptionV3网络结构
def mobileNet(inputs_shape=[224, 224, 3], depth_multiplier=1, dropout=1e-3, classes=1000):
    # Input：这是Keras中的一个函数，用于创建一个输入层。输入层是神经网络的第一层，它接收原始输入数据。
    inputs_images = Input(shape=inputs_shape)

    # 第一层，普通卷积 224,224,3 -> 112,112,32
    x = conv_bn(inputs_images, 32, (3, 3), strides=(2, 2))

    # Conv dw 112,112,32 -> 112,112,64
    x = depthwiseConvBlock(x, 64, depth_multiplier=depth_multiplier, block=1)

    # Conv dw 112,112,64 -> 56,56,128
    x = depthwiseConvBlock(x, 128, depth_multiplier=depth_multiplier, strides=(2, 2), block=2)

    # Conv dw 56,56,128 -> 56,56,128
    x = depthwiseConvBlock(x, 128, depth_multiplier=depth_multiplier, block=3)

    # Conv dw 56,56,128 -> 28,28,256
    x = depthwiseConvBlock(x, 256, depth_multiplier=depth_multiplier, strides=(2, 2), block=4)

    # Conv dw 28,28,256 -> 28,28,256
    x = depthwiseConvBlock(x, 256, depth_multiplier=depth_multiplier, block=5)

    # Conv dw 28,28,256 -> 14,14,512
    x = depthwiseConvBlock(x, 512, depth_multiplier=depth_multiplier, strides=(2, 2), block=6)

    # Conv dw 14,14,512 -> 14,14,512
    x = depthwiseConvBlock(x, 512, depth_multiplier=depth_multiplier, block=7)
    x = depthwiseConvBlock(x, 512, depth_multiplier=depth_multiplier, block=8)
    x = depthwiseConvBlock(x, 512, depth_multiplier=depth_multiplier, block=9)
    x = depthwiseConvBlock(x, 512, depth_multiplier=depth_multiplier, block=10)
    x = depthwiseConvBlock(x, 512, depth_multiplier=depth_multiplier, block=11)

    # Conv dw 14,14,512 -> 7,7,1024
    x = depthwiseConvBlock(x, 1024, depth_multiplier=depth_multiplier, strides=(2, 2), block=12)

    # Conv dw 7,7,1024 -> 7,7,1024
    x = depthwiseConvBlock(x, 1024, depth_multiplier=depth_multiplier, block=13)

    # AvgPool 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)

    # 拍扁reshape，Reshape 层用于改变输入张量的形状而不改变其数据
    x = Reshape((1, 1, 1024), name='reshaped1')(x)

    # dropout 防止过拟合
    x = Dropout(dropout, name='dropout1')(x)

    # FC和softmax
    #x = Dense(classes, activation='softmax', name='FC')(x)
    x = Conv2D(classes, (1, 1),padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    # 创建模型实例，它将输入图像通过定义的层序列处理，最终输出分类结果
    model = Model(inputs_images, x, name='mobileNet')
    return model

# 图像预处理
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
# 主函数
if __name__=='__main__':
    # 模型初始化
    model = mobileNet()

    model.load_weights("mobilenet_1_0_224_tf.h5")
    # 查看模型摘要: 每层的输出形状和模型的总参数数量
    model.summary()
    # 输入需要测试的图片
    images = image.load_img('elephant.jpg', target_size=(224, 224))
    # 将图像转换为数组
    x = image.img_to_array(images)
    # 添加批次维度
    x = np.expand_dims(x, axis=0)
    # 预处理图像
    x = preprocess_input(x)
    # 输出测试结果
    preds = model.predict(x)
    # 对预测结果进行decode
    print('Predicted:', decode_predictions(preds))

