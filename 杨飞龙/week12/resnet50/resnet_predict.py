# 实现resnet50
from __future__ import print_function
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions, preprocess_input

# 定义恒等块（Identity Block），输入输出维度相同
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    ResNet的恒等块（Identity Block）
    :param input_tensor: 输入张量，形状为 (h, w, c)
    :param kernel_size: 卷积核大小
    :param filters: 滤波器数量列表
    :param stage: 阶段编号
    :param block: 块编号
    :return: 输出张量，形状为 (h, w, c)
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

# 定义卷积块（Convolution Block），用于下采样
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    ResNet的卷积块（Convolution Block）
    :param input_tensor: 输入张量，形状为 (h, w, c)
    :param kernel_size: 卷积核大小
    :param filters: 滤波器数量列表
    :param stage: 阶段编号
    :param block: 块编号
    :param strides: 步长
    :return: 输出张量，形状为 (h/2, w/2, c)
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(input_shape=[224, 224, 3], classes=1000):
    """
    构建ResNet50模型
    :param input_shape: 输入形状，形状为 (h, w, c)
    :param classes: 类别数量
    :return: ResNet50模型
    """
    img_input = Input(shape=input_shape)
    # 输入形状: (224, 224, 3)
    # 输出形状: (230, 230, 3)，通过ZeroPadding2D添加3像素的零填充
    x = ZeroPadding2D((3, 3))(img_input)

    # 输入形状: (230, 230, 3)
    # 输出形状: (112, 112, 64)，通过7x7卷积和2x2最大池化
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 输入形状: (112, 112, 64)
    # 输出形状: (56, 56, 256)，通过conv_block和identity_block
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # 输入形状: (56, 56, 256)
    # 输出形状: (28, 28, 512)，通过conv_block和identity_block
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # 输入形状: (28, 28, 512)
    # 输出形状: (14, 14, 1024)，通过conv_block和identity_block
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # 输入形状: (14, 14, 1024)
    # 输出形状: (7, 7, 2048)，通过conv_block和identity_block
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # 输入形状: (7, 7, 2048)
    # 输出形状: (1, 1, 2048)，通过AveragePooling2D
    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    # 输入形状: (1, 1, 2048)
    # 输出形状: (2048,)，通过Flatten
    x = Flatten()(x)

    # 输入形状: (2048,)
    # 输出形状: (classes,)，通过Dense和softmax激活函数
    x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    return model

if __name__ == '__main__':
    model = ResNet50()
    model.summary()
    img_path = 'elephant.jpg'
    # 读取并调整图像大小
    img = image.load_img(img_path, target_size=(224, 224))  # 输入形状: (224, 224)
    x = image.img_to_array(img)  # 转换为数组，形状: (224, 224, 3)
    x = np.expand_dims(x, axis=0)  # 增加批次维度，形状: (1, 224, 224, 3)
    x = preprocess_input(x)  # 预处理输入数据，归一化

    print('Input image shape:', x.shape)  # 打印输入形状
    preds = model.predict(x)  # 进行预测，输出形状: (1, classes)
    print('Predicted:', decode_predictions(preds))  # 解码预测结果
