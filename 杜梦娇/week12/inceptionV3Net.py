import numpy as np
from keras import layers
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, AveragePooling2D
from keras.layers import MaxPooling2D, ZeroPadding2D, Input, Flatten
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.models import Model
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions


def conv_bn(inputs, filters, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    #scale=False 表示在 BatchNormalization 层中，数据只会被规范化，而不会被缩放或平移
    x = Conv2D(filters, kernel_size, name=conv_name, strides=strides, padding=padding, use_bias=False)(inputs)
    x = BatchNormalization(scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


# 定义inceptionV3网络结构
def inceptionV3Net(inputs_shape=[299, 299, 3], classes=1000):
    # Input：这是Keras中的一个函数，用于创建一个输入层。输入层是神经网络的第一层，它接收原始输入数据。
    inputs_images = Input(shape=inputs_shape)

    x = conv_bn(inputs_images, 32, (3, 3), strides=(2, 2), padding='valid')
    x = conv_bn(x, 32, (3, 3),  padding='valid')
    x = conv_bn(x, 64, (3, 3))

    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_bn(x, 80, (1, 1), padding='valid')
    x = conv_bn(x, 192, (3, 3),  padding='valid')

    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # ========================inception模块组1==================================
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inception模块组1  part1 (35 x 35 x 192 -> 35 x 35 x 256(256= 64+64+96+32))
    block1x1 = conv_bn(x, 64, (1, 1))

    block5x5 = conv_bn(x, 48, (1, 1))
    block5x5 = conv_bn(block5x5, 64, (5, 5))

    block3x3 = conv_bn(x, 64, (1, 1))
    block3x3 = conv_bn(block3x3, 96, (3, 3))
    block3x3 = conv_bn(block3x3, 96, (3, 3))

    blockpool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    blockpool = conv_bn(blockpool, 32, (1, 1))

    # concat
    x = layers.concatenate([block1x1, block5x5, block3x3, blockpool], axis=3, name='mixed0')

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inception模块组1 part2 (35 x 35 x 256 -> 35 x 35 x 288(288= 64+64+96+64))
    block1x1 = conv_bn(x, 64, (1, 1))

    block5x5 = conv_bn(x, 48, (1, 1))
    block5x5 = conv_bn(block5x5, 64, (5, 5))

    block3x3 = conv_bn(x, 64, (1, 1))
    block3x3 = conv_bn(block3x3, 96, (3, 3))
    block3x3 = conv_bn(block3x3, 96, (3, 3))

    blockpool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    blockpool = conv_bn(blockpool, 64, (1, 1))

    # concat
    x = layers.concatenate([block1x1, block5x5, block3x3, blockpool], axis=3, name='mixed1')

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inception模块组1 part3 (35 x 35 x 256 -> 35 x 35 x 288(288= 64+64+96+64))
    block1x1 = conv_bn(x, 64, (1, 1))

    block5x5 = conv_bn(x, 48, (1, 1))
    block5x5 = conv_bn(block5x5, 64, (5, 5))

    block3x3 = conv_bn(x, 64, (1, 1))
    block3x3 = conv_bn(block3x3, 96, (3, 3))
    block3x3 = conv_bn(block3x3, 96, (3, 3))

    blockpool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    blockpool = conv_bn(blockpool, 64, (1, 1))

    # concat
    x = layers.concatenate([block1x1, block5x5, block3x3, blockpool], axis=3, name='mixed2')

    #==============================inception模块组2============================================
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inception模块组2 part1 (35 x 35 x 288 -> 17 x 17 x 768(768= 384+96+288))
    block3x3 = conv_bn(x, 384, (3, 3), strides=(2, 2), padding='valid')

    block3x3dbl = conv_bn(x, 64, (1, 1))
    block3x3dbl = conv_bn(block3x3dbl, 96, (3, 3))
    block3x3dbl = conv_bn(block3x3dbl, 96, (3, 3), strides=(2, 2), padding='valid')

    blockpool = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # concat
    x = layers.concatenate([block3x3, block3x3dbl, blockpool], axis=3, name='mixed3')

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inception模块组2 part2 (17 x 17 x 768 -> 17 x 17 x 768(768= 192+192+192+192))
    block1x1 = conv_bn(x, 192, (1, 1))

    block7x7 = conv_bn(x, 128, (1, 1))
    block7x7 = conv_bn(block7x7, 128, (1, 7))
    block7x7 = conv_bn(block7x7, 192, (7, 1))

    block7x7dbl = conv_bn(x, 128, (1, 1))
    block7x7dbl = conv_bn(block7x7dbl, 128, (7, 1))
    block7x7dbl = conv_bn(block7x7dbl, 128, (1, 7))
    block7x7dbl = conv_bn(block7x7dbl, 128, (7, 1))
    block7x7dbl = conv_bn(block7x7dbl, 192, (1, 7))

    blockpool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    blockpool = conv_bn(blockpool, 192, (1, 1))

    # concat
    x = layers.concatenate([block1x1, block7x7, block7x7dbl, blockpool], axis=3, name='mixed4')

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inception模块组2 part3 and 4 (17 x 17 x 768 -> 17 x 17 x 768 -> 17 x 17 x 768)
    for i in range(2):
        block1x1 = conv_bn(x, 192, (1, 1))

        block7x7 = conv_bn(x, 160, (1, 1))
        block7x7 = conv_bn(block7x7, 160, (1, 7))
        block7x7 = conv_bn(block7x7, 192, (7, 1))

        block7x7dbl = conv_bn(x, 160, (1, 1))
        block7x7dbl = conv_bn(block7x7dbl, 160, (7, 1))
        block7x7dbl = conv_bn(block7x7dbl, 160, (1, 7))
        block7x7dbl = conv_bn(block7x7dbl, 160, (7, 1))
        block7x7dbl = conv_bn(block7x7dbl, 192, (1, 7))

        blockpool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        blockpool = conv_bn(blockpool, 192, (1, 1))

        # concat
        x = layers.concatenate([block1x1, block7x7, block7x7dbl, blockpool], axis=3, name='mixed' + str(5 + i))

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inception模块组2 part5 (17 x 17 x 768 -> 17 x 17 x 768)
    block1x1 = conv_bn(x, 192, (1, 1))

    block7x7 = conv_bn(x, 192, (1, 1))
    block7x7 = conv_bn(block7x7, 192, (1, 7))
    block7x7 = conv_bn(block7x7, 192, (7, 1))

    block7x7dbl = conv_bn(x, 192, (1, 1))
    block7x7dbl = conv_bn(block7x7dbl, 192, (7, 1))
    block7x7dbl = conv_bn(block7x7dbl, 192, (1, 7))
    block7x7dbl = conv_bn(block7x7dbl, 192, (7, 1))
    block7x7dbl = conv_bn(block7x7dbl, 192, (1, 7))

    blockpool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    blockpool = conv_bn(blockpool, 192, (1, 1))

    # concat
    x = layers.concatenate([block1x1, block7x7, block7x7dbl, blockpool], axis=3, name='mixed7')

    #==============================inception模块组3============================================
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inception模块组3 part1 (17 x 17 x 768 -> 8 x 8 x 1280)
    block3x3 = conv_bn(x, 192, (1, 1))
    block3x3 = conv_bn(block3x3, 320, (3, 3), strides=(2, 2), padding='valid')

    block7x7x3 = conv_bn(x, 192, (1, 1))
    block7x7x3 = conv_bn(block7x7x3, 192, (1, 7))
    block7x7x3 = conv_bn(block7x7x3, 192, (7, 1))
    block7x7x3 = conv_bn(block7x7x3, 192, (3, 3), strides=(2, 2), padding='valid')

    blockpool = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # concat
    x = layers.concatenate([block3x3, block7x7x3, blockpool], axis=3, name='mixed8')

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # inception模块组3 part2和3 (8 x 8 x 1280 -> 8 x 8 x 2048 -> 8 x 8 x 2048)
    for i in range(2):
        block1x1 = conv_bn(x, 320, (1, 1))

        block3x3 = conv_bn(x, 384, (1, 1))
        blockblock3x3x1 = conv_bn(block3x3, 384, (1, 3))
        blockblock3x3x2 = conv_bn(block3x3, 384, (3, 1))
        # concat
        block3x3 = layers.concatenate([blockblock3x3x1, blockblock3x3x2], axis=3, name='mixed9_' + str(i))

        block3x3dbl = conv_bn(x, 448, (1, 1))
        block3x3dbl = conv_bn(block3x3dbl, 384, (3, 3))
        block3x3dbl_1 = conv_bn(block3x3dbl, 384, (1, 3))
        block3x3dbl_2 = conv_bn(block3x3dbl, 384, (3, 1))
        # concat
        block3x3dbl = layers.concatenate([block3x3dbl_1, block3x3dbl_2], axis=3)


        blockpool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        blockpool = conv_bn(blockpool, 192, (1, 1))

        # concat
        x = layers.concatenate([block1x1, block3x3, block3x3dbl, blockpool], axis=3, name='mixed' + str(9 + i))

    # 池化层
    x = GlobalAveragePooling2D(name='average_pool')(x)
    # 全连接层:进行分类
    x = Dense(classes, activation='softmax', name='FC')(x)
    # 创建模型实例，它将输入图像通过定义的层序列处理，最终输出分类结果
    model = Model(inputs_images, x, name='inceptionV3Net')

    return model

# 图像预处理
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
# 主函数
if __name__=='__main__':
    # 模型初始化
    model = inceptionV3Net()

    model.load_weights("inception_v3_weights_tf_dim_ordering_tf_kernels.h5")
    # 查看模型摘要: 每层的输出形状和模型的总参数数量
    model.summary()
    # 输入需要测试的图片
    images = image.load_img('elephant.jpg', target_size=(299, 299))
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

