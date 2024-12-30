import tensorflow as tf
from tensorflow.keras import layers, Model

"""
inputs: 输入张量，形状为 (batch_size, height, width, channels)。
num_classes: 输出类别数，默认为 1000（ImageNet 数据集的类别数）。
is_training: 是否处于训练模式，默认为 True。
dropout_keep_prob: Dropout 层保留的概率，默认为 0.5。
spatial_squeeze: 是否对输出进行空间维度的压缩，默认为 True。
scope: 名称作用域，默认为 'vgg_16'。
"""


def vgg_16(inputs,
           num_classes=1000,
           dropout_keep_prob=0.5,
           is_training=True):
    # 使用 Sequential 模型的方式来组织 VGG 网络
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1_1')(inputs)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_1')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)

    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_1')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_2')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)

    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_1')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_2')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool4')(x)

    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_1')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_2')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_3')(x)
    x = layers.MaxPooling2D((2, 2), name='pool5')(x)

    # 模拟全连接层
    x = layers.Conv2D(4096, (7, 7), padding='valid', activation='relu', name='fc6')(x)
    x = layers.Dropout(1 - dropout_keep_prob, name='dropout6')(x, training=is_training)

    x = layers.Conv2D(4096, (1, 1), padding='valid', activation='relu', name='fc7')(x)
    x = layers.Dropout(1 - dropout_keep_prob, name='dropout7')(x, training=is_training)

    x = layers.Conv2D(num_classes, (1, 1), padding='valid', activation=None, name='fc8')(x)

    # 展平输出
    x = tf.squeeze(x, axis=[1, 2], name='fc8/squeezed')

    return x


# 示例输入和构建模型
input_tensor = tf.keras.Input(shape=(224, 224, 3))

output_tensor = vgg_16(input_tensor, num_classes=1000)

# 定义模型
model = Model(inputs=input_tensor, outputs=output_tensor)
# 打印模型
model.summary()
model.save('vgg16_model.h5')  # 保存为 .h5 文件
model.save('vgg16_saved_model')  # 保存为 TensorFlow 原生 SavedModel 格式

