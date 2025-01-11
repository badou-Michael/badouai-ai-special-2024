# -*- coding: utf-8 -*-
# time: 2024/11/20 23:16
# file: MobileNet.py
# author: flame
import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.layers import DepthwiseConv2D,Input,Activation,Dropout,Reshape,BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling2D,Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K

''' 整体逻辑：实现一个基于MobileNet的图像分类模型，包括ReLU6激活函数、卷积块、深度可分离卷积块和预处理输入函数。模型加载预训练权重并进行图像分类预测。 '''

def relu6(x):
    ''' 实现ReLU6激活函数，限制输出在0到6之间。 '''
    return K.relu(x, max_value=6)

def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    ''' 定义一个标准的卷积块，包括卷积层、批量归一化和ReLU6激活。 '''
    ''' 参数: inputs - 输入张量, filters - 卷积核数量, kernel - 卷积核大小, strides - 步长 '''
    ''' 返回: 经过卷积、批量归一化和ReLU6激活后的张量 '''
    x = Conv2D(filters, kernel, padding='same', use_bias=False, strides=strides, name='conv1')(inputs)
    ''' 使用Conv2D层进行卷积操作，设置填充方式为'same'，不使用偏置项，步长为指定值，名称为'conv1'。 '''
    x = BatchNormalization(name='conv1_bn')(x)
    ''' 应用批量归一化，名称为'conv1_bn'。 '''
    return Activation(relu6, name='conv1_relu6')(x)
    ''' 应用ReLU6激活函数，名称为'conv1_relu6'。 '''

def _depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
    ''' 定义一个深度可分离卷积块，包括深度卷积、批量归一化、ReLU6激活、逐点卷积、批量归一化和ReLU6激活。 '''
    ''' 参数: inputs - 输入张量, pointwise_conv_filters - 逐点卷积核数量, depth_multiplier - 深度乘数, strides - 步长, block_id - 块ID '''
    ''' 返回: 经过深度可分离卷积、批量归一化和ReLU6激活后的张量 '''
    x = DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_multiplier, strides=strides, use_bias=False, name='conv_dw_%d' % block_id)(inputs)
    ''' 使用DepthwiseConv2D层进行深度卷积操作，设置填充方式为'same'，深度乘数为指定值，步长为指定值，不使用偏置项，名称为'conv_dw_{block_id}'。 '''
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    ''' 应用批量归一化，名称为'conv_dw_{block_id}_bn'。 '''
    x = Activation(relu6, name='conv_dw_%d_relu6' % block_id)(x)
    ''' 应用ReLU6激活函数，名称为'conv_dw_{block_id}_relu6'。 '''
    x = Conv2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1), name='conv_pw_%d' % block_id)(x)
    ''' 使用Conv2D层进行逐点卷积操作，设置卷积核大小为(1, 1)，填充方式为'same'，不使用偏置项，步长为(1, 1)，名称为'conv_pw_{block_id}'。 '''
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    ''' 应用批量归一化，名称为'conv_pw_{block_id}_bn'。 '''
    return Activation(relu6, name='conv_pw_%d_relu6' % block_id)(x)
    ''' 应用ReLU6激活函数，名称为'conv_pw_{block_id}_relu6'。 '''

def MobileNet(input_shape=[224, 224, 3], depth_multiplier=1, dropout=1e-3, classes=1000):
    ''' 定义MobileNet模型，包括输入层、多个卷积块和深度可分离卷积块、全局平均池化、全连接层和Softmax激活。 '''
    ''' 参数: input_shape - 输入图像尺寸, depth_multiplier - 深度乘数, dropout - Dropout比率, classes - 分类类别数 '''
    ''' 返回: 构建好的MobileNet模型 '''
    img_input = Input(shape=input_shape)
    ''' 定义输入层，形状为指定的input_shape。 '''
    x = _conv_block(img_input, 32, strides=(2, 2))
    ''' 应用第一个卷积块，输出32个特征图，步长为(2, 2)。 '''
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)
    ''' 应用第一个深度可分离卷积块，输出64个特征图。 '''
    x = _depthwise_conv_block(x, 128, depth_multiplier, strides=(2, 2), block_id=2)
    ''' 应用第二个深度可分离卷积块，输出128个特征图，步长为(2, 2)。 '''
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)
    ''' 应用第三个深度可分离卷积块，输出128个特征图。 '''
    x = _depthwise_conv_block(x, 256, depth_multiplier, strides=(2, 2), block_id=4)
    ''' 应用第四个深度可分离卷积块，输出256个特征图，步长为(2, 2)。 '''
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)
    ''' 应用第五个深度可分离卷积块，输出256个特征图。 '''
    x = _depthwise_conv_block(x, 512, depth_multiplier, strides=(2, 2), block_id=6)
    ''' 应用第六个深度可分离卷积块，输出512个特征图，步长为(2, 2)。 '''
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    ''' 应用第七个深度可分离卷积块，输出512个特征图。 '''
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    ''' 应用第八个深度可分离卷积块，输出512个特征图。 '''
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    ''' 应用第九个深度可分离卷积块，输出512个特征图。 '''
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    ''' 应用第十个深度可分离卷积块，输出512个特征图。 '''
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)
    ''' 应用第十一个深度可分离卷积块，输出512个特征图。 '''
    x = _depthwise_conv_block(x, 1024, depth_multiplier, strides=(2, 2), block_id=12)
    ''' 应用第十二个深度可分离卷积块，输出1024个特征图，步长为(2, 2)。 '''
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)
    ''' 应用第十三个深度可分离卷积块，输出1024个特征图。 '''
    x = GlobalAveragePooling2D()(x)
    ''' 应用全局平均池化层，将特征图压缩为固定长度的向量。 '''
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    ''' 将池化后的特征图重塑为(1, 1, 1024)的形状。 '''
    x = Dropout(dropout, name='dropout')(x)
    ''' 应用Dropout层，防止过拟合。 '''
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    ''' 应用1x1卷积层，输出类别数个特征图。 '''
    x = Activation('softmax', name='act_softmax')(x)
    ''' 应用Softmax激活函数，输出概率分布。 '''
    x = Reshape((classes,), name='reshape_2')(x)
    ''' 将输出重塑为(classes,)的形状。 '''
    inputs = img_input
    model = Model(inputs, x, name='mobilenet_%g_%g_%s' % (depth_multiplier, classes, input_shape))
    ''' 构建模型，输入为img_input，输出为x，名称为'mobilenet_{depth_multiplier}_{classes}_{input_shape}'。 '''
    model.load_weights("mobilenet_1_0_224_tf.h5")
    ''' 加载预训练权重。 '''
    return model
    ''' 返回构建好的模型。 '''

def preprocess_input(x):
    ''' 预处理输入图像，将其归一化到[-1, 1]范围。 '''
    ''' 参数: x - 输入图像数组 '''
    ''' 返回: 归一化后的图像数组 '''
    x /= 255
    ''' 将图像数组除以255，使其值在[0, 1]范围内。 '''
    x -= 0.5
    ''' 减去0.5，使值在[-0.5, 0.5]范围内。 '''
    x *= 2.
    ''' 乘以2，使值在[-1, 1]范围内。 '''
    return x
    ''' 返回归一化后的图像数组。 '''

if __name__ == '__main__':
    model = MobileNet()
    ''' 创建MobileNet模型实例。 '''
    model.summary()
    ''' 打印模型的结构摘要。 '''

    img = image.load_img('bike.jpg', target_size=(224, 224))
    ''' 加载图像文件，调整大小为(224, 224)。 '''
    x = image.img_to_array(img)
    ''' 将图像转换为数组。 '''
    x = np.expand_dims(x, axis=0)
    ''' 在数组的第一个维度增加一个维度，使其形状为(1, 224, 224, 3)。 '''
    x = preprocess_input(x)
    ''' 对图像数组进行预处理。 '''

    preds = model.predict(x)
    ''' 使用模型进行预测。 '''
    print(np.argmax(preds))
    ''' 打印预测结果中概率最高的类别索引。 '''
    print('Predicted', decode_predictions(preds, 1))
    ''' 打印预测结果及其对应的类别标签。 '''
