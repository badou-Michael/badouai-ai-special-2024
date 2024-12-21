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
              depth_multiplier=1,
              dropout=1e-3,
              classes=1000):
    """
    构建MobileNet模型。
    
    参数:
        input_shape: 输入图像的形状，默认为[224, 224, 3]（RGB图像）。
        depth_multiplier: 深度乘数，控制depthwise卷积层中的通道数。
        dropout: Dropout层的比例，用于防止过拟合。
        classes: 分类的数量，默认为1000（ImageNet数据集的分类数）。
        
    返回:
        model: 完整的MobileNet模型。
    """
    img_input = Input(shape=input_shape)
    # 初始卷积层，减少空间尺寸并增加通道数至32
    # 224,224,3 -> 112,112,32
    x = _conv_block(img_input, 32, strides=(2, 2))
    # 第一个深度可分离卷积块，保持空间尺寸不变，增加通道数至64
    # 112,112,32 -> 112,112,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)
    # 第二个深度可分离卷积块，减少空间尺寸至56x56，增加通道数至128
    # 112,112,64 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier,
                              strides=(2, 2), block_id=2)
    # 再次应用深度可分离卷积块，保持空间尺寸不变
    # 56,56,128 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)
    # 减少空间尺寸至28x28，增加通道数至256
    # 56,56,128 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier,
                              strides=(2, 2), block_id=4)
    # 再次应用深度可分离卷积块，保持空间尺寸不变
    # 28,28,256 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)
    # 减少空间尺寸至14x14，增加通道数至512
    # 28,28,256 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier,
                              strides=(2, 2), block_id=6)
    # 应用多个深度可分离卷积块，保持空间尺寸不变，通道数为512
    # 14,14,512 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)
    # 减少空间尺寸至7x7，增加通道数至1024
    # 14,14,512 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)
    # 全局平均池化层，将特征图的空间尺寸缩小到1x1
    # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    # Dropout层，用于防止过拟合
    x = Dropout(dropout, name='dropout')(x)
    # 1x1卷积层，用于分类任务
    x = Conv2D(classes, (1, 1),padding='same', name='conv_preds')(x)
    # Softmax激活函数，生成最终的概率分布
    x = Activation('softmax', name='act_softmax')(x)
    # 最终输出层，展平成一维向量
    x = Reshape((classes,), name='reshape_2')(x)
    # 创建模型
    inputs = img_input
    model = Model(inputs, x, name='mobilenet_1_0_224_tf')
    # 加载预训练权重
    model_name = 'mobilenet_1_0_224_tf.h5'
    model.load_weights(model_name)

    return model

def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    """
    标准卷积块，包括卷积、批量归一化和ReLU6激活函数。
    
    参数:
        inputs: 输入张量。
        filters: 卷积核的数量。
        kernel: 卷积核大小，默认为(3, 3)。
        strides: 步长，默认为(1, 1)。
        
    返回:
        x: 输出张量。
    """
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    """
    深度可分离卷积块，包括depthwise卷积、批量归一化、ReLU6激活、pointwise卷积等步骤。
    参数:
        inputs: 输入张量。
        pointwise_conv_filters: pointwise卷积后的通道数。
        depth_multiplier: 控制depthwise卷积中每个输入通道生成的输出通道数。
        strides: 步长，默认为(1, 1)。
        block_id: 块的标识符，用于命名。
    返回:
        x: 输出张量。
    """
    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

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
    return K.relu(x, max_value=6)

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__':
    model = MobileNet(input_shape=(224, 224, 3))
    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    preds = model.predict(x)
    print(np.argmax(preds))
    print('Predicted:', decode_predictions(preds,1))  # 只显示top1

