'''
from __future__ import print_function：
这个导入是为了确保在Python 2.x版本中使用print函数时，其行为与Python 3.x版本一致。


from __future__ import absolute_import 是 Python 2.x 中的一个导入语句，
用于确保在 Python 2.x 环境下使用绝对导入，而不是相对导入。
在 Python 3.x 中，所有的导入默认都是绝对导入，因此这个语句在 Python 3.x 中是多余的。

'''



from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

'''
from keras.models import Model：导入Model类，用于创建和训练整个模型。
from keras import layers：导入Keras的layers模块，包含了构建神经网络所需的各种层。
from keras.layers import ...：从keras.layers模块中导入了多个具体的层类型，
如Input, Dense, Conv2D等，这些都是构建神经网络时常用的层。
'''
from keras.models import Model
from keras import layers
from keras.layers import Activation,Dense,Input,BatchNormalization,Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D
# TensorFlow 的 Keras 模块中导入了 get_source_inputs 函数
from tensorflow.keras.utils import get_source_inputs
'''
这两行代码首先导入了 TensorFlow 库，然后使用 tf.keras.backend.set_image_data_format 
函数将图像数据格式设置为 channels_last。这意味着在处理图像数据时，通道（颜色通道）将被放在最后一个维度。
例如，对于一个 RGB 图像，其形状将是 (height, width, 3)。
'''
import tensorflow as tf
tf.keras.backend.set_image_data_format('channels_last')  # 或 'channels_first'
'''
导入 get_file 函数，用于下载文件并保存到本地路径
导入 Keras 后端，通常用于访问低级的 TensorFlow 操作。
导入 decode_predictions 函数，用于将模型预测结果解码为人类可读的标签。
导入 image 模块，其中包含图像预处理相关的函数。
导入 load_img 和 img_to_array 函数，分别用于加载图像和将图像转换为 NumPy 数组。
'''
from tensorflow.keras.utils import get_file
from tensorflow.keras import backend as K
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
'''
构建一个带有批量归一化（Batch Normalization）和激活函数（ReLU）的二维卷积层conv2d_bn的函数
x: 输入张量。
filters: 卷积核的数量。
num_row: 卷积核的高度。
num_col: 卷积核的宽度。
strides: 卷积步长，默认为 (1,1)。
padding: 填充方式，默认为 'same'，表示输出与输入大小相同。
name: 层的名称，如果提供，将用于卷积层和批量归一化层的名称。
'''
def conv2d_bn(x,filters,num_row,num_col,strides = (1,1), padding = 'same', name = None):
    if name is not None:
        bn_name = name + 'bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = Conv2D(filters, (num_row,num_col),strides = strides, padding = padding,
               use_bias = False, name = conv_name)(x)
    x = BatchNormalization(scale = False, name = bn_name)(x)
    x = Activation('relu', name = name)(x)
    return x
def InceptionV3(input_shape = [299,299,3], classes = 1000):
    img_input = Input(shape = input_shape)
    # 提取32个特征
    x = conv2d_bn(img_input,32,3,3,strides = (2,2),padding = 'valid')
    # conv 149*149*32  ===》conv padded 147*147*32
    x = conv2d_bn(x,32, 3,3,padding = 'valid')
    # conv padded 147*147*32 ===》pool 144*1476*64
    x = conv2d_bn(x,64,3,3)
    # 池化后输入尺寸x = (147-3)/2+1 =73
    x = MaxPooling2D((3,3), strides = (2,2))(x)
    # 卷积提取80个特征（参照InveptionV3的网络结构）
    x = conv2d_bn(x,80,1,1,padding = 'valid')
    # （73-3）/1+1 =71
    x = conv2d_bn(x,192,3,3,padding = 'valid')
    # (71-3)/2+1 = 35
    x = MaxPooling2D((3,3),strides = (2,2))(x)
    # -----------------------------
    #     Block1 35*35
    # -----------------------------
    # block1 part1
    # 35 * 35 * 192 ====》 35 * 35 * 256
    branch1x1 = conv2d_bn(x,64,1,1)

    branch5x5 = conv2d_bn(x,48,1,1)
    branch5x5 = conv2d_bn(branch5x5,64,5,5)

    branch3x3dbl = conv2d_bn(x,64,1,1)
    branch3x3dbl = conv2d_bn(branch3x3dbl,96,3,3)
    branch3x3dbl = conv2d_bn(branch3x3dbl,96,3,3)
    # 使用3x3的平均池化层，步长为1，填充方式为'same'，以保持输出的空间维度不变
    branch_pool = AveragePooling2D((3,3),strides =(1,1),padding = 'same')(x)
    # 在池化后的输出上应用1x1卷积核，以提取特征并减少维度。
    branch_pool = conv2d_bn(branch_pool,32,1,1)
    '''
    将所有分支的输出沿着通道轴（axis=3）进行拼接，得到最终的输出特征图。
    这种拼接方式使得网络能够同时利用不同尺度的特征:64+64+96+32 = 256
    '''
    x = layers.concatenate([branch1x1,branch5x5,branch3x3dbl,branch_pool],
                           axis =3,name ='mixed0')

    # block1 part2
    # 35 * 35 * 256 ====》 35 * 35 * 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    # 使用3x3的平均池化层，步长为1，填充方式为'same'，以保持输出的空间维度不变
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    # 在池化后的输出上应用1x1卷积核，以提取特征并减少维度。
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    '''
    将所有分支的输出沿着通道轴（axis=3）进行拼接，得到最终的输出特征图。
    这种拼接方式使得网络能够同时利用不同尺度的特征:64+64+96+64 = 288
    '''
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                           axis=3, name='mixed1')

    # block1 part3
    # 35 * 35 * 288 ====》 35 * 35 * 288
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    # 使用3x3的平均池化层，步长为1，填充方式为'same'，以保持输出的空间维度不变
    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    # 在池化后的输出上应用1x1卷积核，以提取特征并减少维度。
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    '''
    将所有分支的输出沿着通道轴（axis=3）进行拼接，得到最终的输出特征图。
    这种拼接方式使得网络能够同时利用不同尺度的特征:64+64+96+64 = 288
    '''
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                           axis=3, name='mixed2')

    # -----------------------------
    #     Block2 17*17
    # -----------------------------
    # block2 part1
    # 35 * 35 * 288 ====》 17 * 17 * 768
    branch3x3 =conv2d_bn(x,384,3,3,strides = (2,2), padding ='valid')
    branch3x3dbl = conv2d_bn(x,64,1,1)
    branch3x3dbl = conv2d_bn(branch3x3dbl,96,3,3)
    branch3x3dbl = conv2d_bn(branch3x3dbl,96,3,3,strides = (2,2), padding = 'valid')
    # 使用3x3的平均池化层，步长为1，填充方式为'same'，以保持输出的空间维度不变
    branch_pool = MaxPooling2D((3,3),strides = (2,2))(x)
    # 在池化后的输出上应用1x1卷积核，以提取特征并减少维度。
    '''
    将所有分支的输出沿着通道轴（axis=3）进行拼接，得到最终的输出特征图。
    这种拼接方式使得网络能够同时利用不同尺度的特征:384+96+288 = 768
    '''
    x = layers.concatenate([branch3x3,branch3x3dbl,branch_pool],
                           axis =3,name ='mixed3')
    '''
    Inception 模块包含了多个分支，每个分支都使用了不同大小的卷积核来提取特征，
    然后将这些特征合并。这种设计允许网络同时学习不同尺度和形状的特征，从而提高模型的性能
    '''
    # block2 part2
    # 17 * 17 * 768 ====》 17 * 17 * 768
    # 这个分支使用1 x1的卷积核，卷积核数量为192。
    # 1x1卷积主要用于降维和增加非线性，同时允许网络在不增加感受野的情况下学习特征的压缩表示。
    branch1x1 = conv2d_bn(x,192,1,1)
    # 这个分支首先使用1x1 卷积核降维到128个通道，然后通过1x7和7x1的卷积核组合来扩展感受野。
    # 这种分解的卷积核设计减少了参数数量和计算量，同时能够捕捉更复杂的特征。
    branch7x7 = conv2d_bn(x,128,1,1)
    branch7x7 = conv2d_bn(branch7x7,128,1,7)
    branch7x7 = conv2d_bn(branch7x7,192,7,1)
    '''
    这个分支通过两次应用 1x7 和 7x1 的卷积核来进一步扩展感受野。它首先使用 1x1 卷积核降维到 128 个通道，
    然后通过一系列分解的卷积核来提取更复杂的特征。这种设计进一步增加了网络的深度和特征提取能力。
    '''
    branch7x7dbl = conv2d_bn(x,128,1,1)
    branch7x7dbl = conv2d_bn(branch7x7dbl,128,7,1)
    branch7x7dbl = conv2d_bn(branch7x7dbl,128,1,7)
    branch7x7dbl = conv2d_bn(branch7x7dbl,128,7,1)
    branch7x7dbl = conv2d_bn(branch7x7dbl,192,1,7)

    branch_pool = AveragePooling2D((3,3),strides =(1,1), padding = 'same')(x)
    branch_pool = conv2d_bn(branch_pool,192,1,1)
    #192+192+192+192 =768
    x  = layers.concatenate([branch1x1,branch7x7,branch7x7dbl,branch_pool],
                            axis =3, name = 'mixed4')

    # block2 part3 and part4
    # 17 * 17 * 768 ====》 17 * 17 * 768  ====》 17 * 17 * 768
    for i in range (2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                               axis=3, name='mixed'+str(5+i))
    # block2 part5
    # 17 * 17 * 768 ====》 17 * 17 * 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3,3),strides =(1,1), padding = 'same')(x)
    branch_pool = conv2d_bn(branch_pool,192,1,1)
    #192+192+192+192 =768
    x  = layers.concatenate([branch1x1,branch7x7,branch7x7dbl,branch_pool],
                            axis =3, name = 'mixed7')
    # -----------------------------
    #     Block3 8*8
    # -----------------------------
    # block3 part1
    # 17 * 17 * 768 ====》 8 * 8 * 1280
    branch3x3 = conv2d_bn(x,192,1,1)
    branch3x3 = conv2d_bn(branch3x3,320,3,3,strides =(2,2),padding = 'valid')

    branch7x7x3 = conv2d_bn(x,192,1,1)
    branch7x7x3 = conv2d_bn(branch7x7x3,192,1,7)
    branch7x7x3 = conv2d_bn(branch7x7x3,192,7,1)
    branch7x7x3 = conv2d_bn(branch7x7x3,192,3,3,strides = (2,2), padding = 'valid')

    branch_pool = MaxPooling2D((3,3),strides = (2,2))(x)
    #320+192+768 =1280
    x = layers.concatenate([branch3x3,branch7x7x3,branch_pool],axis = 3, name = 'mixed8')

    # Block3 part2 and part3
    # 8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x,320,1,1)

        branch3x3 = conv2d_bn(x,384,1,1)
        branch3x3_1 = conv2d_bn(branch3x3,384,1,3)
        branch3x3_2 = conv2d_bn(branch3x3,384,3,1)
        branch3x3 = layers.concatenate([branch3x3_1,branch3x3_2],axis = 3,name = 'mixde9_'+str(i) )

        branch3x3dbl = conv2d_bn(x,448,1,1)
        branch3x3dbl = conv2d_bn(branch3x3dbl,384,3,3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl,384,1,3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl,384,3,1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1,branch3x3dbl_2],axis =3)

        branch_pool = AveragePooling2D((3,3),strides=(1,1), padding = 'same')(x)
        branch_pool = conv2d_bn(branch_pool,192,1,1)
        x = layers.concatenate([branch1x1,branch3x3,branch3x3dbl,branch_pool],axis =3,
                               name = 'mixed' + str(9+i))
    #平均池化后的全连接，参照inceptionV3网络结构模块组后面步骤
    x = GlobalAveragePooling2D(name = 'avg_pool')(x)
    # 神经网络中用于分类任务的最后一层。这个层将输入特征映射到指定数量的输出类别，
    # 并使用softmax激活函数来输出每个类别的概率
    x = Dense(classes,activation = 'softmax',name = 'predictions')(x)

    inputs = img_input
    model = Model(inputs, x, name = 'inceptionV3' )
    return model

def preprocess_input(x):
    x /= 255.0
    x -= 0.5
    x *= 2.0
    return x
if __name__ == '__main__':
    model = InceptionV3()
    model.load_weights('../task/week12/inceptionV3_tf/inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
    # model.load_weights('../task/week12/inceptionV3_tf/inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
    #                    by_name=False, skip_mismatch=False)
    img_path = '../task/week12/inceptionV3_tf/elephant.jpg'
    '''
    load_img 函数加载图像，并将其大小调整为 (299,299)。
    img_to_array 函数将 PIL 图像对象转换为 NumPy 数组。
    np.expand_dims 在数组前面增加一个维度，以匹配模型输入的批次维度。
    preprocess_input 函数对图像进行预处理，使其符合预训练模型的输入要求。
    '''
    img = load_img(img_path,target_size = (299,299))
    x = img_to_array(img)
    x = np.expand_dims(x,axis = 0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print ('predict:',decode_predictions(preds))



