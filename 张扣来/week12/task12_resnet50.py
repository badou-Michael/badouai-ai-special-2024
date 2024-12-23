# resnet50网络部分
'''
from __future__ import print_function：
这个导入是为了确保在Python 2.x版本中使用print函数时，其行为与Python 3.x版本一致。
from keras import layers：导入Keras的layers模块，包含了构建神经网络所需的各种层。
from keras.layers import ...：从keras.layers模块中导入了多个具体的层类型，
如Input, Dense, Conv2D等，这些都是构建神经网络时常用的层。
'''
from __future__ import print_function
import numpy as np
from keras import layers
from keras.layers import Input
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import Activation,BatchNormalization,Flatten
'''
from keras.models import Model：导入Model类，用于创建和训练整个模型。
from keras.preprocessing import image：导入Keras的图像预处理模块。
import keras.backend as K：导入Keras的后端模块，用于访问底层的TensorFlow或其他后端的操作。
from keras.utils.data_utils import get_file：导入用于下载数据集的工具函数。
from keras.applications.imagenet_utils import decode_predictions：
导入用于将模型预测的类别ID解码为人类可读的标签的函数。
from keras.applications.imagenet_utils import preprocess_input：
导入用于预处理输入数据以匹配ImageNet数据集的函数。
'''
from keras.models import Model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import keras.backend as K
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    '''
    用于构建卷积层（Conv2D）和批量归一化层（BatchNormalization）的名称
    stage：这个变量表示网络中的一个阶段或层级。例如，在残差网络中，stage 可能代表不同的残差块组。
    block：这个变量表示在特定阶段中的一个块或子块。
    conv_name_base 将根据 stage 和 block 的值生成一个唯一的名称
    '''
    conv_name_base  = 'res' + str(stage) + block + '_branch'#卷积层名称
    bn_name_base  = 'bn' + str(stage) + block + '_branch'#批量归一化层名称
    # Conv2D是Keras中的一个层，用于实现卷积神经网络中的二维卷积操作。
    # 这种类型的卷积层通常用于处理图像数据，因为它能够捕捉到图像中的局部空间特征。
    # filters1为卷积核数量，input_tensor 为输入张量
    # 构建残差网络（ResNet）中残差块过程
    x = Conv2D(filters1,(1,1), name = conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name = bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    # 对输入张量x应用了一个1x1卷积层。filters2是这个卷积层的滤波器数量。这个1x1卷积通常用于降维或升维
    x = Conv2D(filters2,kernel_size,padding = 'same', name = conv_name_base + '2b')(x)
    # 紧接着1x1卷积层的是批量归一化层，它有助于加速训练过程，减少内部协变量偏移，并可能提高模型的性能。
    # 这里的name参数与1x1卷积层的名称相同，这有助于在复杂的网络中追踪和管理这些层。
    x = BatchNormalization(name = bn_name_base +'2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1,1), name = conv_name_base + '2c')(x)
    x = BatchNormalization(name = bn_name_base + '2c')(x)
    '''
    这一行代码实现了残差连接，即将原始输入 input_tensor 和经过1x1卷积及批量归一化处理后的输出 x 相加。
    这是残差网络的核心思想， 它允许网络学习到的残差映射（输入和输出之间的差异）通过这个连接直接传递，
    有助于解决深度网络中的梯度消失问题。
    '''
    x = layers.add([x,input_tensor])
    x = Activation('relu')(x)
    return x
def conv_block(input_tensor, kernel_size, filters, stage, block, strides = (2,2)):
    filters1, filters2, filters3, = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1,(1,1), strides = strides, name =conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name = bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2,kernel_size, padding = 'same', name = conv_name_base + '2b')(x)
    x = BatchNormalization(name= bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3,(1,1), name = conv_name_base + '2c' )(x)
    x = BatchNormalization(name = bn_name_base + '2c')(x)
    shortcut = Conv2D(filters3,(1,1),strides = strides, name = conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name = bn_name_base + '1')(shortcut)

    x = layers.add([x,shortcut])
    x = Activation('relu')(x)
    return x
def ResNet50(input_shape = [224,224,3], classes = 1000):
    # Input是Keras模型定义中的一个函数，用于指定网络输入数据的形状
    img_input = Input(shape = input_shape)
    # 对输入图像 img_input 应用了零填充。ZeroPadding2D 是 Keras 中的一个层，
    # 用于在输入数据的边界周围添加零值填充。
    x = ZeroPadding2D((3,3))(img_input)
    x = Conv2D(64,(7,7),strides = (2,2), name = 'conv1')(x)
    x = BatchNormalization(name = 'bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3),strides = (2,2))(x)
    # 第2层残差块组
    x= conv_block(x,3,[64,64,256], stage = 2, block = 'a',strides = (1,1))
    x = identity_block(x,3,[64,64,256],stage = 2, block = 'b')
    x = identity_block(x,3,[64,64,256],stage = 2, block = 'c')
    # 第3层残差块组
    x = conv_block(x,3,[128,128,512], stage = 3, block = 'a')
    x = identity_block(x,3,[128,128,512],stage = 3,block = 'b')
    x = identity_block(x,3,[128,128,512],stage = 3,block = 'c')
    x = identity_block(x,3,[128,128,512],stage = 3,block = 'd')
    # 第4层残差块组
    x = conv_block(x,3,[256,256,1024], stage =4, block = 'a')
    x = identity_block(x,3,[256,256,1024], stage = 4, block = 'b')
    x = identity_block(x,3,[256,256,1024], stage = 4, block = 'c')
    x = identity_block(x,3,[256,256,1024], stage = 4, block = 'd')
    x = identity_block(x,3,[256,256,1024], stage = 4, block = 'e')
    x = identity_block(x,3,[256,256,1024], stage = 4, block = 'f')
    # 第5层残差块组
    x = conv_block(x,3,[512,512,2048],stage = 5, block = 'a')
    x = identity_block(x,3,[512,512,2048], stage = 5, block = 'b')
    x = identity_block(x,3,[512,512,2048], stage = 5, block = 'c')

    # 均值池化
    x = AveragePooling2D((7,7), name = 'avg_pool')(x)
    # 多维拍扁
    x = Flatten()(x)
    '''
    全连接层将展平后的一维向量映射到类别数量 classes 的输出空间。a
    ctivation='softmax' 表示使用 softmax 激活函数，它会将输出转换为概率分布，每个类别对应一个概率值。
    name='fc1000' 为这个层指定了一个名称
    '''
    x = Dense(classes,activation = 'softmax', name = 'fc1000')(x)
    model = Model(img_input, x, name = 'resnet50')
    model.load_weights('../task/week12/resnet50_tf/resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                       by_name=True, skip_mismatch=True)
    return model
if __name__ == '__main__':
    # 模型初始化
    model = ResNet50()
    # 打印出模型的结构，包括每层的名称、输出形状和参数数量
    model.summary()
    # 图像加载和预处理
    img_path = '../task/week12/resnet50_tf/elephant.jpg'
    # img_path = '../task/week12/resnet50_tf/bike.jpg'
    '''
    load_img 函数加载图像，并将其大小调整为 (224,224)。
    img_to_array 函数将 PIL 图像对象转换为 NumPy 数组。
    np.expand_dims 在数组前面增加一个维度，以匹配模型输入的批次维度。
    preprocess_input 函数对图像进行预处理，使其符合预训练模型的输入要求。
    '''
    img = load_img(img_path, target_size = (224,224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    print('input image shape:',x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))
