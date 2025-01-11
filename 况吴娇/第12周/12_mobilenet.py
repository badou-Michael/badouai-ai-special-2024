#-------------------------------------------------------------#
#   MobileNet的网络部分
#-------------------------------------------------------------#
import warnings
import numpy as np

from keras.preprocessing import image

from keras.models import Model
from keras.layers import DepthwiseConv2D,Input,Activation,Dropout,Reshape,BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling2D,Conv2D
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K


def MobileNet(input_shape=[224,224,3],
              depth_multiplier=1, #深度乘数，用于控制网络的“宽度”，即过滤器的数量。
              dropout=1e-3,#1×10^−3 Dropout 率为 0.1%。
              classes=1000): #classes: 输出类别数，默认为1000（ImageNet数据集的类别数）。


    img_input = Input(shape=input_shape)

    # 224,224,3 -> 112,112,32
    x = _conv_block(img_input, 32, strides=(2, 2))
    '''
    当使用 padding='same' 时，理论上输出的空间维度（高度和宽度）应该与输入的空间维度相同，前提是步长（strides）为 (1, 1)。
    在 _conv_block 中，高度和宽度发生了变化，这主要是因为卷积层的步长（strides）设置为 (2, 2)，导致了下采样。
    
    '''

    # 112,112,32 -> 112,112,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)
    '''padding='same' 时，Keras 会自动调整填充，使得输出尺寸为输入尺寸除以步长的结果。这样可以简化计算，并确保输出尺寸符合预期'''
    # 112,112,64 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier,
                              strides=(2, 2), block_id=2)
    # 56,56,128 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # 56,56,128 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier,
                              strides=(2, 2), block_id=4)
    
    # 28,28,256 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # 28,28,256 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier,
                              strides=(2, 2), block_id=6)
    
    # 14,14,512 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 14,14,512 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x) #将这个 1, 1, 1024 的输出重塑为 (1, 1, 1024) 的张量。
    x = Dropout(dropout, name='dropout')(x) #Dropout 层会随机将一部分输出置零，以减少过拟合。假设 dropout 率为 0.1，那么大约有 10% 的特征值会被置零。

    '''
    在您之前提供的 MobileNet 代码中，"FC/sl" 对应的部分是：
x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
x = Activation('softmax', name='act_softmax')(x)
x = Reshape((classes,), name='reshape_2')(x)
这里，Conv2D(classes, (1, 1), padding='same', name='conv_preds') 模拟了全连接层的行为，
通过1x1卷积将特征图的通道数从1024减少到类别数（例如1000）。然后，Activation('softmax') 应用 Softmax 函数，将输出转换为概率分布。
最后，Reshape((classes,), name='reshape_2') 将输出重塑为一维数组，每个元素对应一个类别的概率。
   '''
    x = Conv2D(classes, (1, 1),padding='same', name='conv_preds')(x)
    '''
    我们使用一个 1x1 的卷积核进行逐点卷积，将 1024 个特征通道减少到模型的输出类别数。假设我们的模型是为一个有 1000 个类别的分类任务设计的，那么这里的 Conv2D 层将把特征通道数从 1024 减少到 1000。
    '''
    x = Activation('softmax', name='act_softmax')(x)
    '''
    我们应用 softmax 激活函数将这些原始预测值转换为概率分布。softmax 函数确保所有输出值都在 0 到 1 之间，并且总和为 1，这样就可以将它们解释为概率。
    '''
    x = Reshape((classes,), name='reshape_2')(x)
    ''''
    Reshape((classes,)) 中的逗号是必要的，因为它表示一个形状元组（tuple），即使这个元组只包含一个元素。
    Reshape((classes,)) 操作是将模型输出的张量重塑为一个二维张量，其中第一个维度是批次大小（batch size），第二个维度是类别数（classes）。
如果批次大小是 1（即一次处理一张图片），那么 Reshape((classes,)) 操作后的张量形状将是 (1, classes)，也就是 (1, 1000)。这意味着有一个样本，每个样本有 1000 个类别的预测分数。
如果批次大小大于 1，比如一次处理 32 张图片，那么输出张量的形状将是 (batch_size, classes)，也就是 (32, 1000)。这意味着有 32 个样本，每个样本有 1000 个类别的预测分数。
所以，Reshape((classes,)) 操作后的张量形状是：
如果批次大小为 1，则为 (1, classes)
如果批次大小为 N，则为 (N, classes)
这里的 classes 是列数，每个列代表一个类别的预测分数。
    '''
    #将这个 1, 1, 1000 的输出重塑为 (1000,) 的一维数组，每个元素对应一个类别的概率。
    #最后，使用 Reshape 层将输出张量的形状从 (batch_size, classes) 重塑为 (batch_size,)，使其成为一维数组，每个元素对应一个类别的概率。
    '''
    全局平均池化层 GlobalAveragePooling2D() 会计算每个特征通道在整个 7x7 网格上的平均值。这样，每个 1024 个特征通道都会被压缩成一个单一的数值，表示该通道在整个 7x7 网格上的平均激活值。
    
    输入: 7, 7, 1024
    输出: 1, 1, 1024
    '''
    inputs = img_input

    model = Model(inputs, x, name='mobilenet_1_0_224_tf')
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)

    return model

def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)): #filters: 卷积层中的过滤器（卷积核）数量。
    #kernel=(3, 3): 卷积核的尺寸，默认为3x3。strides=(1, 1): 卷积的步长，默认为(1, 1)，表示卷积核每次移动一个像素。
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters,##pointwise_conv_filters: 这是逐点卷积（1x1卷积）层的过滤器数量。
                          depth_multiplier=1, strides=(1, 1), block_id=1): #depth_multiplier=1: 深度乘数，用于控制深度可分离卷积中每个输入通道的输出通道数。默认值为1，意味着每个输入通道产生一个输出通道。
#block_id=1: 用于标识当前块的ID，主要用于生成层的名称，以便在模型中区分不同的块。
    x = DepthwiseConv2D((3, 3),
                        padding='same', #name='conv_dw_%d' % block_id: 层的名称，使用block_id来区分不同的块
                        depth_multiplier=depth_multiplier,
                        #深度乘数，用于控制输出通道数。
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs) #name='conv_dw_%d' % block_id: 层的名称，使用block_id来区分不同的块
    '''
    DepthwiseConv2D 是 Keras 中的一个层，用于执行深度可分离卷积（Depthwise Separable Convolution）。这种卷积是一种优化技术，它将标准的卷积分解为两个较小的操作：
    深度卷积（Depthwise Convolution）和逐点卷积（Pointwise Convolution）。这种分解可以显著减少模型的计算量和参数数量，而不会牺牲太多准确性，
    特别适合于计算资源受限的环境，如移动设备或嵌入式系统。
    '''
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def relu6(x):
    return K.relu(x, max_value=6)#K.relu: Keras后端的ReLU激活函数，最大值为6。

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x #将输入图像归一化到[-1, 1]范围。

if __name__ == '__main__':
    model = MobileNet(input_shape=(224, 224, 3))

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    #predict 方法是 Keras 模型对象 model 的一个内置函数。当你创建一个 Keras 模型并编译它之后，模型对象会包含几种方法，其中之一就是 predict 方法。
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds,1))
    # 只显示top1
    '''image.load_img: 从指定路径加载图像，并将其调整到目标大小 (224, 224)。
 image.img_to_array: 将 PIL 图像转换为 NumPy 数组。
 np.expand_dims: 在第一个维度（批次维度）上扩展数组，因为模型预测需要批次维度。
 preprocess_input: 预处理输入图像，使其符合模型的输入要求。这通常包括归一化和减去均值等操作。
 
 model.predict 方法返回模型的输出。
 np.argmax(preds): 从预测结果中找到概率最高的类别索引。
 decode_predictions(preds, 1): 解码预测结果，这里只显示概率最高的一个预测结果（top-1）。
 
 
 '''
