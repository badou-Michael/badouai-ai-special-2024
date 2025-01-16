# -------------------------------------------------------------#
#   ResNet50的网络部分（TensorFlow 2版本）
# -------------------------------------------------------------#

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.layers import Activation, BatchNormalization, Flatten, Add
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as K
from tensorflow.keras.utils import get_file
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
import numpy as np

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    实现ResNet50的身份块(identity block)

    参数：
        input_tensor: 输入张量
        kernel_size: 卷积核大小
        filters: 整数序列, 定义卷积核的数量
        stage: 整数, 用于生成层的名称
        block: 字符串, 用于生成层的名称

    返回：
        输出张量
    """
    filters1, filters2, filters3 = filters

    conv_base = 'res' + str(stage) + block + '_branch'
    bn_base = 'bn' + str(stage) + block + '_branch'

    # 第一层
    x = Conv2D(filters1, (1, 1), name=conv_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_base + '2a')(x)
    x = Activation('relu')(x)

    # 第二层
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_base + '2b')(x)
    x = BatchNormalization(name=bn_base + '2b')(x)
    x = Activation('relu')(x)

    # 第三层
    x = Conv2D(filters3, (1, 1), name=conv_base + '2c')(x)
    x = BatchNormalization(name=bn_base + '2c')(x)

    # 将输入张量和主路径的输出相加
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    实现ResNet50的卷积块(conv block)

    参数：
        input_tensor: 输入张量
        kernel_size: 卷积核大小
        filters: 整数序列，定义卷积核的数量
        stage: 整数，用于生成层的名称
        block: 字符串，用于生成层的名称
        strides: 步长

    返回：
        输出张量
    """
    filters1, filters2, filters3 = filters

    conv_base = 'res' + str(stage) + block + '_branch'
    bn_base = 'bn' + str(stage) + block + '_branch'

    # 主路径
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_base + '2b')(x)
    x = BatchNormalization(name=bn_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_base + '2c')(x)
    x = BatchNormalization(name=bn_base + '2c')(x)

    # 短连接（shortcut）路径
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_base + '1')(shortcut)

    # 将shortcut和主路径的输出相加
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(input_shape=(224, 224, 3), classes=1000):
    """
    构建ResNet50模型

    参数：
        input_shape: 输入图像的形状
        classes: 分类数量

    返回：
        ResNet50模型
    """
    img_input = Input(shape=input_shape)

    # 初始卷积和池化层
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 堆叠多个残差块
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # 平均池化层
    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    # 全连接层
    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc' + str(classes))(x)

    # 创建模型
    model = Model(inputs=img_input, outputs=x, name='resnet50')

    return model

if __name__ == '__main__':
    # 创建ResNet50模型
    model = ResNet50()
    # 输出模型结构
    model.summary()

    # 加载预训练权重
    weights_path = './Course_CV/Week12/demo/resnet50_tf/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path)

    # 加载并预处理图像
    img_path = './Course_CV/Week12/demo/resnet50_tf/elephant.jpg'  # 图像路径
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('输入图像的形状:', x.shape)

    # 进行预测
    preds = model.predict(x)
    print('预测结果:', decode_predictions(preds, top=5)[0])