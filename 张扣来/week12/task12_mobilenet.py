# mobilenet网络部分
import warnings
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import DepthwiseConv2D,Input,Activation,Dropout,Reshape
from tensorflow.keras.layers import BatchNormalization,GlobalAveragePooling2D,GlobalMaxPooling2D,Conv2D
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras import backend as K
'''
input_shape: 输入图像的形状，默认为 [224, 224, 3]，表示输入图像的尺寸为 224x224 像素，且有 3 个颜色通道（RGB）。
depth_multiplier: 深度乘数，用于控制深度卷积层中每个输入通道生成的输出通道数量。
默认为 1，表示每个输入通道生成一个输出通道。增加 depth_multiplier 可以增加模型的容量，但也会增加计算量。
dropout: Dropout 层的丢弃率，默认为 1e-3（即 0.1%）。Dropout 是一种正则化技术，用于减少过拟合的风险。
在训练过程中，它会随机丢弃一定比例的神经元。
classes: 输出层的类别数量，默认为 1000。这表示模型将用于一个有 1000 个类别的分类任务，例如 ImageNet 数据集。
'''
def MobileNet(input_shape = [224,224,3],depth_multiplier = 1,dropout = 1e-3, classes = 1000):
    img_input = Input(shape = input_shape)

    # 224 x 224 x 3 ===》 112 x 112 x 32  conv_block卷积核为3，3
    x = _conv_block(img_input,32,strides = (2,2))
    # 112 x 112 x 32 ===》 112 x 112 x 64  _depthwise_conv_block卷积模式为same
    # _depthwise_conv_block卷积里面包括3x3卷积核和1x1卷积核
    x = _depthwise_conv_block(x,64, depth_multiplier, block_id = 1)

    # 112 x 112 x 64 ===》 56 x 56 x 128
    x = _depthwise_conv_block(x,128,depth_multiplier,strides = (2,2),block_id =2)
    # 56 x 56 x 128 ===》 56 x 56 x 128
    x = _depthwise_conv_block(x,128,depth_multiplier,block_id =3)

    # 56 x 56 x 128 ===》 28 x 28 x 256
    x = _depthwise_conv_block(x,256,depth_multiplier,strides = (2,2),block_id =4)
    # 28 x 28 x 256 ===》 28 x 28 x 256
    x = _depthwise_conv_block(x,256,depth_multiplier,block_id=5)

    # 28 x 28 x 256 ===》 14 x 14 x 512
    x = _depthwise_conv_block(x,512,depth_multiplier,strides = (2,2),block_id =6)
    # 14 x 14 x 512 ===》 14 x 14 x 512
    x = _depthwise_conv_block(x,512,depth_multiplier,block_id=7)
    x = _depthwise_conv_block(x,512,depth_multiplier,block_id=8)
    x = _depthwise_conv_block(x,512,depth_multiplier,block_id=9)
    x = _depthwise_conv_block(x,512,depth_multiplier,block_id=10)
    x = _depthwise_conv_block(x,512,depth_multiplier,block_id=11)

    # 14 x 14 x 512 ===》 7 x 7 x 1024
    x = _depthwise_conv_block(x,1024,depth_multiplier,strides = (2,2),block_id =12)
    # 7 x 7 x 1024 ===》 7 x 7 x 1024
    x = _depthwise_conv_block(x,1024,depth_multiplier,block_id=13)
    # 7 x 7 x 1024 ===》 1 x 1 x 1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1,1,1024), name = 'reshape_1')(x)
    x = Dropout(dropout, name = 'dropout')(x)

    x = Conv2D(classes,(1,1), padding = 'same', name = 'conv_preds')(x)
    x = Activation('softmax', name = 'act_softmax')(x)
    x = Reshape((classes,), name = 'reshape_2')(x)

    inputs = img_input
    model = Model(inputs, x, name ='mobilenet_1_0_224' )
    model_name = '../task/week12/mobilenet/mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)
    return model

def _conv_block(inputs,filters,kernel = (3,3), strides = (1,1)):
    x = Conv2D(filters, kernel, padding = 'same' ,use_bias = False, strides = strides, name = 'conv1')(inputs)
    x = BatchNormalization(name = 'conv1_bn')(x)
    # 注意Activation传递的是函数名称，而不是relu6()去调用这个函数
    x = Activation(relu6, name = 'conv1_relu')(x)
    return x
'''
深度卷积(Depthwise Convolution）和逐点卷积(Pointwise Convolution)
inputs: 输入张量。
pointwise_conv_filters: 逐点卷积的卷积核数量。
depth_multiplier: 深度卷积的深度乘数，默认为 1。它控制每个输入通道生成的输出通道数量。
strides: 卷积步长，默认为 (1,1)。
block_id: 块的编号，用于生成层的名称。
'''
def _depthwise_conv_block(inputs, pointwise_conv_filters, depth_multiplier = 1, strides = (1,1), block_id = 1):
    '''
    使用 DepthwiseConv2D 层对输入进行深度卷积。每个输入通道独立地应用一个卷积核。
    (3,3): 卷积核尺寸。
    padding='same': 确保输出尺寸与输入尺寸相同。
    depth_multiplier: 深度乘数。
    strides: 卷积步长。
    use_bias=False: 不使用偏置，因为后面有批量归一化层。
    name='conv_dw_%d'% block_id: 层的名称，根据块编号动态生成。
    '''
    x = DepthwiseConv2D((3,3), padding = 'same', depth_multiplier = depth_multiplier, strides = strides,
                       use_bias = False, name = 'conv_dw_%d'% block_id)(inputs)
    # 对深度卷积的输出进行批量归一化处理
    x = BatchNormalization(name = 'conv_dw_%d_bn' % block_id)(x)
    # Activation常用函数有激活函数名称和层名称，层指定一个名称可以在模型的可视化和调试过程中更容易地识别该层
    x = Activation(relu6, name = 'conv_dw_%d_relu'% block_id)(x)
    '''
    使用 Conv2D 层进行逐点卷积，实际上是一个 1x1 的卷积操作，用于对深度卷积的输出进行线性组合，以生成所需数量的输出通道。
    参数：
    pointwise_conv_filters: 卷积核数量。
    (1,1): 卷积核尺寸。
    padding='same': 确保输出尺寸与输入尺寸相同。
    use_bias=False: 不使用偏置。
    strides=(1,1): 卷积步长。
    name='conv_pw_%d' % block_id: 层的名称。
    '''
    x = Conv2D(pointwise_conv_filters,(1,1),padding = 'same', use_bias = False,strides = (1,1),
               name = 'conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name = 'conv_pw_%d_bn' % block_id)(x)
    x = Activation(relu6, name = 'conv_pw_%s_relu'% block_id)(x)
    return x
def relu6(x):
    return K.relu(x, max_value = 6)
'''
简单的图像预处理函数，通常用于在将图像数据输入到深度学习模型之前对其进行标准化处理。
这个函数的目的是将图像像素值从 [0, 255] 范围转换到 [-1, 1] 范围。*2后落在【-1，1】中
'''
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
if __name__ == '__main__':
    model = MobileNet(input_shape = (224,224,3))
    img_path = '../task/week12/mobilenet/elephant.jpg'
    img = load_img(img_path, target_size = (224,224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    print('input image shape:',x.shape)
    preds = model.predict(x)
    print(np.argmax(preds))
    print ('predicted:',decode_predictions(preds,1))
