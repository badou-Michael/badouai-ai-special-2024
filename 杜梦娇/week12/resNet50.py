import numpy as np
from keras import layers
from keras.layers import Input
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, Dense, AveragePooling2D
from keras.layers import Activation, BatchNormalization, Flatten
from keras.models import Model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input

# 定义卷积块
def convBlock(inputs, kernel_size, filters, stage, block, strides=(2, 2)):
    # inputs：输入（tensor），即前一层的输出。
    # kernel_size：卷积核的大小，通常是一个元组，表示卷积核的高和宽。
    # filters：一个包含三个元素的元组，分别表示三个卷积层的过滤器（filter）数量。
    # stage：表示当前块所在的阶段，用于命名。
    # block：表示当前块的编号，用于命名。
    # strides：卷积层的步长，默认为(2, 2)，表示在高和宽方向上每次移动两个像素。

    # 它将filters这个元组或列表中的三个元素分别赋值给filters1、filters2和filters3三个变量,这些变量代表不同卷积层的过滤器（或称为卷积核）数量
    filters1, filters2, filters3 = filters
    # 构建了一个基础名称，用于后续卷积层的命名
    conv_name_base = 'res' + str(stage) + block + '_branch'
    # 用于后续标准化层的命名
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 定义网络层
    x = Conv2D(filters1, (1, 1),strides=strides, name=conv_name_base + '2a')(inputs)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 定义并联分支，即快捷连接（shortcut connection）
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(inputs)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    # 输入激活函数中的输入，即两个分支输出之和
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def identityBlock(inputs, kernel_size, filters, stage, block):
    # inputs：输入（tensor），即前一层的输出。
    # kernel_size：卷积核的大小，通常是一个元组，表示卷积核的高和宽。
    # filters：一个包含三个元素的元组，分别表示三个卷积层的过滤器（filter）数量。
    # stage：表示当前块所在的阶段，用于命名。
    # block：表示当前块的编号，用于命名。

    # 它将filters这个元组或列表中的三个元素分别赋值给filters1、filters2和filters3三个变量,这些变量代表不同卷积层的过滤器（或称为卷积核）数量
    filters1, filters2, filters3 = filters
    # 构建了一个基础名称，用于后续卷积层的命名
    conv_name_base = 'res' + str(stage) + block + '_branch'
    # 用于后续标准化层的命名
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 定义网络层
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(inputs)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 输入激活函数中的输入，即两个分支输出之和
    x = layers.add([x, inputs])
    x = Activation('relu')(x)
    return x



# 定义resnet50网络结构
def resNet50(inputs_shape=[224, 224, 3], classes=1000):
    # Input：这是Keras中的一个函数，用于创建一个输入层。输入层是神经网络的第一层，它接收原始输入数据。
    inputs_images = Input(shape=inputs_shape)
    # zeropad层
    x = ZeroPadding2D((3, 3))(inputs_images)
    # conv层提取公共特征
    x = Conv2D(64, (7, 7),strides=(2, 2), name='conv1')(x)
    # 标准化
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    # 最大池化层
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 残差网络块1
    x = convBlock(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identityBlock(x, 3, [64, 64, 256], stage=2, block='b')
    x = identityBlock(x, 3, [64, 64, 256], stage=2, block='c')

    # 残差网络块2
    x = convBlock(x, 3, [128, 128, 512], stage=3, block='a')
    x = identityBlock(x, 3, [128, 128, 512], stage=3, block='b')
    x = identityBlock(x, 3, [128, 128, 512], stage=3, block='c')
    x = identityBlock(x, 3, [128, 128, 512], stage=3, block='d')

    # 残差网络块3
    x = convBlock(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identityBlock(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identityBlock(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identityBlock(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identityBlock(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identityBlock(x, 3, [256, 256, 1024], stage=4, block='f')

    # 残差网络块4
    x = convBlock(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identityBlock(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identityBlock(x, 3, [512, 512, 2048], stage=5, block='c')

    # 均值池化层
    x = AveragePooling2D((7, 7), name='average_pool')(x)
    # Flatten:将多维输入展平成一维
    x = Flatten()(x)
    # 全连接层:进行分类
    x = Dense(classes, activation='softmax', name='FC')(x)
    # 创建模型实例，它将输入图像通过定义的层序列处理，最终输出分类结果
    model = Model(inputs_images, x, name='resNet50')

    # 加载预训练权重模型
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    return model

# 主函数
if __name__=='__main__':
    # 模型初始化
    model = resNet50()
    # 查看模型摘要: 每层的输出形状和模型的总参数数量
    model.summary()
    # 输入需要测试的图片
    # img_path = 'elephant.jpg'
    images_path = 'bike.jpg'
    images = image.load_img(images_path, target_size=(224, 224))
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

