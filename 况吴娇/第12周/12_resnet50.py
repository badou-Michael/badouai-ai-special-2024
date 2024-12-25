'''
在ResNet50中，确实存在两种类型的残差块：Identity Block和Convolutional Block。这两种块的主要区别在于Shortcut Connection上是否进行了卷积操作。
Identity Block：在这种块中，输入和输出的维度是相同的，因此Shortcut Connection不需要进行卷积操作，直接将输入与输出相加
Convolutional Block：在这种块中，输入和输出的维度可能不同。为了使输入与输出的维度匹配，Shortcut Connection会使用1x1卷积进行维度转换。这种卷积操作的主要目的是调整输入的通道数，使其与输出的通道数一致
#-------------------------------------------------------------#
#   ResNet50的网络部分
#-------------------------------------------------------------#
'''
#-------------------------------------------------------------#
#   ResNet50的网络部分
#-------------------------------------------------------------#
from __future__ import print_function
##确保在Python 2.x中使用Python 3.x的print函数。
import numpy as np
from keras import layers #导入Keras的层模块，用于构建神经网络。

from keras.layers import Input
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D
'''
MaxPooling2D：二维最大池化层。它用于降低卷积层输出的空间维度，同时保留最重要的特征。最大池化通过取池化窗口内的最大值来实现。
AveragePooling2D：二维平均池化层。与最大池化类似，但它取池化窗口内的平均值，而不是最大值。
'''
from keras.layers import Activation,BatchNormalization,Flatten
'''
Activation：激活层用于在神经网络中引入非线性。它将一个非线性函数应用于输入数据。常见的激活函数包括 ReLU（线性修正单元）、sigmoid、tanh 等。
BatchNormalization：批量归一化层用于改善神经网络训练过程的稳定性。它通过对每个特征减去均值并除以标准差来规范化输入数据，从而减少内部协变量偏移。
Flatten：展平层用于将多维输入一维化，通常用于卷积层和全连接层之间。它将卷积层的三维输出（高度、宽度、通道数）展平为一维数组，以便可以被全连接层处理。
'''
from keras.models import Model
from keras.preprocessing import image #image 模块包含用于图像预处理的函数，如加载图像、将图像转换为数组、保存图像等。
import keras.backend as K#K 是 Keras 后端的别名。它提供了一系列低级操作，允许你直接与后端（如 TensorFlow、Theano 或 CNTK）交互。
# 使用 Keras 后端可以执行张量操作、变量定义等
from keras.utils.data_utils import get_file ##get_file 函数用于下载资源（如果尚未下载）并返回本地路径。它通常用于下载预训练模型的权重文件。
from keras.applications.imagenet_utils import decode_predictions
#decode_predictions 函数用于将模型预测的类别编码转换为人类可读的标签。它通常用于 ImageNet 预训练模型的输出解释。
from keras.applications.imagenet_utils import preprocess_input ##提供ImageNet数据集的工具函数，如解码预测结果和预处理输入。

#定义身份块（Identity Block）

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch' #创建卷积层和批量归一化层的名称基础，这些名称用于区分模型中的不同层。
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    ##使用 1×1 的卷积核的目的通常不是为了减少通道数，而是为了增加或减少通道数，同时保持特征图的空间维度不变。
    x = BatchNormalization(name=bn_name_base + '2a')(x) ##name 参数允许你为这个层指定一个名称，这在模型中用于标识不同的层，方便调试和可视化。
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)

    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base  + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=[224,224,3],classes=1000):
    img_input =Input(shape=input_shape)  #定义模型的输入层，输入形状为input_shape。
    x=ZeroPadding2D((3, 3))(img_input)
    x=Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

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

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)#Flatten层用于将多维的输入一维化。这通常在卷积神经网络（CNN）中使用，
    # 用于将卷积层或池化层输出的多维特征图转换为一维特征向量，以便可以被全连接层（Dense层）处理。
    # Flatten层将输入张量x转换为一维张量。具体来说，它将输入张量的所有非批量维度（即除了第一个维度之外的所有维度）展平成一个单一的维度。
    #假设x是一个形状为(batch_size, height, width, channels)的四维张量，经过Flatten()层处理后，输出张量的形状将变为(batch_size, height * width * channels)。这样，每个样本的所有特征都被展平到一个单一的一维数组中。
    x = Dense(classes, activation='softmax', name='fc1000')(x)
    '''
    创建了一个Keras模型实例：Model：这是Keras中用于定义模型的类。它允许你指定模型的输入和输出。
     img_input：这是模型的输入层，通常是一个Input层的实例，定义了输入数据的形状。
     x：这是模型的输出层，通常是经过一系列层（如卷积层、池化层、全连接层等）处理后的最终输出。
     name='resnet50'：这是模型的名称，用于标识和引用模型。
    通过这种方式，你可以将输入层和输出层连接起来，定义一个完整的模型架构。
    '''
    model = Model(img_input, x, name='resnet50')
    ##加载预训练权重
    model.load_weights("resnet50_weights_tf_dim_ordering_tf_kernels.h5")
    '''
    load_weights：这是Model类的一个方法，用于加载保存在HDF5文件中的模型权重。
    "resnet50_weights_tf_dim_ordering_tf_kernels.h5"：这是包含预训练权重的文件名。这些权重通常是在大规模数据集（如ImageNet）上训练得到的。
    加载预训练权重可以显著提高模型的性能，特别是在数据量有限的情况下。预训练权重为模型提供了一个良好的初始化，减少了训练时间，并有助于模型更快地收敛。
    '''
    return model

if __name__=='__main__':

    model=ResNet50() ## 注意这里使用了括号来创建一个实例
    model.summary()#打印模型的摘要，包括每层的类型、输出形状和参数数量。这有助于验证模型结构和调试。

    '''
    summary() 方法的输出是一个表格，每一行代表模型中的一个层。表格的第一列显示层的名称和类型，
    第二列显示输出形状，第三列显示参数数量。表格的底部显示总参数数量、可训练参数数量和不可训练参数数量。
    Model: "resnet50"
    __________________________________________________________________________________________________
   Layer (type)                    Output Shape         Param #     Connected to    
    '''
    img_path = 'bike.jpg'
    img = image.load_img(img_path, target_size=(224, 224)) #使用Keras的load_img函数加载图像，并将其调整为目标尺寸224x224像素。这是ResNet50模型的期望输入尺寸。
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x) #preprocess_input 函数来自于 Keras 的 applications 模块，具体来说，它通常与特定的预训练模型相关联
    #在 Keras 中，preprocess_input 函数的作用是：归一化像素值：将图像的像素值从 [0, 255] 缩放到 [0, 1] 或其他特定范围;
    #调整均值和标准差：减去训练数据集的均值并除以标准差，以使输入数据的分布与训练时使用的数据一致。




    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))



    '''
    img_path = 'elephant.jpg'：指定要加载的图像路径。这里使用elephant.jpg作为示例。
img = image.load_img(img_path, target_size=(224, 224))：使用Keras的load_img函数加载图像，并将其调整为目标尺寸224x224像素。这是ResNet50模型的期望输入尺寸。
x = image.img_to_array(img)：将PIL图像对象转换为NumPy数组。
x = np.expand_dims(x, axis=0)：增加一个维度，以匹配模型输入的批量维度。模型期望输入的形状为(batch_size, height, width, channels)。
x = preprocess_input(x)：对输入图像进行预处理，使其与训练时的图像具有相同的均值和标准差。这是为了确保模型能够正确地处理输入数据。

print('Input image shape:', x.shape)：打印预处理后的输入图像的形状，以验证其符合模型的输入要求。
preds = model.predict(x)：使用模型对预处理后的图像进行预测。predict方法返回预测结果，通常是一个概率分布。
print('Predicted:', decode_predictions(preds))：使用decode_predictions函数将模型的输出解码为类别标签和相应的概率。这使得预测结果更易于理解和解释
    '''

