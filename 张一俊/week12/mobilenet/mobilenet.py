#   MobileNet的网络结构实现

import numpy as np

from keras.preprocessing import image

from keras.models import Model
from keras.layers import DepthwiseConv2D,Input,Activation,Dropout,Reshape,BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling2D,Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


def MobileNet(input_shape=[224,224,3], depth_multiplier=1, dropout=1e-3, classes=1000):
    """
    创建MobileNet模型
    :param input_shape: 输入图像的形状
    :param depth_multiplier: 深度乘子，控制每个深度卷积层的输出通道数
    :param dropout: Dropout率
    :param classes: 输出类别数
    :return: 模型
    """

    img_input = Input(shape=input_shape)

    # 224x224x3 -> 112x112x32
    x = _conv_block(img_input, 32, strides=(2, 2))

    # 112x112x32 -> 112x112x64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    # 112x112x64 -> 56x56x128
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)

    # 56x56x128 -> 56x56x128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # 56x56x128 -> 28x28x256
    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)

    # 28x28x256 -> 28x28x256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # 28x28x256 -> 14x14x512
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)

    # 14x14x512 -> 14x14x512
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)

    # 14x14x512 -> 14x14x512 (Repeated depthwise blocks)
    for i in range(8, 13):
        x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=i)

    # 14x14x512 -> 7x7x1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)

    # 7x7x1024 -> 1x1x1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 全局平均池化 -> 全连接层
    x = GlobalAveragePooling2D()(x)  # 将特征图尺寸降到1x1
    x = Reshape((1, 1, 1024), name='reshape_1')(x)  # 改变形状以适应卷积层
    x = Dropout(dropout, name='dropout')(x)  # Dropout层防止过拟合
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)  # 全连接层（通过1x1卷积）
    x = Activation('softmax', name='act_softmax')(x)  # softmax激活函数，输出最终分类结果
    x = Reshape((classes,), name='reshape_2')(x)  # 重塑为一维向量（输出类别）

    inputs = img_input

    model = Model(inputs, x, name='mobilenet_1_0_224_tf')

    # 加载预训练权重
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)

    return model


def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    """
    卷积块，用于标准卷积操作和批量归一化。
    :param inputs: 输入层。
    :param filters: 卷积核数目。
    :param kernel: 卷积核大小，默认为(3, 3)。
    :param strides: 步长，默认为(1, 1)。
    :return: 激活后的卷积层输出。
    """
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    """
    深度可分离卷积块，包含深度卷积和逐点卷积（1x1卷积）。
    :param inputs: 输入层。
    :param pointwise_conv_filters: 逐点卷积（1x1卷积）的输出通道数。
    :param depth_multiplier: 深度乘子，控制每个深度卷积输出的通道数。
    :param strides: 步长，默认为(1, 1)。
    :param block_id: 当前块的ID，用于命名卷积层。
    :return: 激活后的深度卷积层输出。
    """
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_multiplier, strides=strides, use_bias=False, name=f'conv_dw_{block_id}')(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1), name=f'conv_pw_{block_id}')(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def relu6(x):
    # 限制最大值为6的ReLU输出
    return K.relu(x, max_value=6)


def preprocess_input(x):
    # 预处理输入图像
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__':
    model = MobileNet(input_shape=(224, 224, 3))

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)  # 增加批量维度
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    # 进行预测
    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds, 1))  # 只显示top1

