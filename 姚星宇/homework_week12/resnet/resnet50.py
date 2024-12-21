#-------------------------------------------------------------#
#   ResNet50的网络部分
#-------------------------------------------------------------#
from __future__ import print_function

import numpy as np
from keras import layers
from keras.layers import Input
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import Activation,BatchNormalization,Flatten
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    实现ResNet的恒等块（Identity Block），不改变输入张量的空间尺寸
    参数:
        input_tensor: 输入张量
        kernel_size: 中间卷积层使用的核大小
        filters: 卷积层的过滤器数量，是一个长度为3的列表
        stage: 整数，当前阶段编号，用于命名
        block: 字符串/'a','b'等，当前块的标识符，用于命名
    返回:
        x: 输出张量
    """
    filters1, filters2, filters3 = filters
    print("filters1:", filters1)
    print("filters2:", filters2)
    print("filters3:", filters3)
    # 构建图形名称
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 第一层：1x1卷积减少维度
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    # 第二层：使用指定kernel_size的卷积层
    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # 第三层：1x1卷积恢复维度
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    # 将输入与输出相加
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    实现ResNet的卷积块（Convolutional Block），会改变输入张量的空间尺寸 
    参数:
        input_tensor: 输入张量
        kernel_size: 中间卷积层使用的核大小
        filters: 卷积层的过滤器数量，是一个长度为3的列表
        stage: 整数，当前阶段编号，用于命名
        block: 字符串/'a','b'等，当前块的标识符，用于命名
        strides: 整数或元组/列表，指定卷积步长，默认(2, 2)
    返回:
        x: 输出张量
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # 第一层：1x1卷积减少维度，同时通过strides参数改变空间尺寸
    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    # 第二层：使用指定kernel_size的卷积层
    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # 第三层：1x1卷积恢复维度
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    # shortcut分支：1x1卷积调整输入张量的通道数和空间尺寸以匹配主路径
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
    # 将输入与输出相加
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(input_shape=[224,224,3],classes=1000):
    """
    创建一个ResNet50模型实例
    参数:
        input_shape: 输入图像的形状，默认为[224, 224, 3]（RGB图像）
        classes: 分类的数量，默认为1000（ImageNet数据集的分类数）
    返回:
        model: 完整的ResNet50模型
    """
    # 定义输入层
    img_input = Input(shape=input_shape)
    # 零填充层，使图像边缘增加3个像素，保持卷积后的尺寸不变
    x = ZeroPadding2D((3, 3))(img_input)
    # 初始卷积层，7x7卷积，64个滤波器，步长为2，然后是批量归一化、激活和最大池化
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    # 第二阶段：构建三个块，使用64个滤波器
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    # 第三阶段：构建四个块，使用128个滤波器
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    # 第四阶段：构建六个块，使用256个滤波器
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')
    # 第五阶段：构建三个块，使用512个滤波器
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    # 平均池化层
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    # 展平层和全连接层，最后使用softmax激活函数进行分类
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc1000')(x)
    # 创建模型
    model = Model(img_input, x, name='resnet50')
    # 加载预训练权重
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    return model

if __name__ == '__main__':
    # 如果脚本直接执行，则构建模型并打印其摘要
    model = ResNet50()
    model.summary()
    # 测试用例：加载图片并进行预测
    # img_path = 'elephant.jpg'
    img_path = 'bike.jpg'
    # 加载图像并调整尺寸为224x224
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('Input image shape:', x.shape)
    # 使用模型进行预测
    preds = model.predict(x) 
    # 打印预测结果
    print('Predicted:', decode_predictions(preds))
