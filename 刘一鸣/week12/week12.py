'''
第十二周作业：
实现resnet、inception、mobilenet
'''

#1、resnet
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# 残差块（Residual Block）
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    # 主分支卷积
    x = layers.Conv2D(filters, kernel_size, stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, stride, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # 如果输入和输出的通道数不一致，或者空间尺寸不一致，需要调整 shortcut
    if shortcut.shape[-1] != filters:  # 通道数不匹配
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # 如果空间尺寸不一致，调整尺寸
    if shortcut.shape[1] != x.shape[1] or shortcut.shape[2] != x.shape[2]:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # 将主分支和shortcut相加
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x

# 构建 ResNet 模型
def create_resnet_model(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # 初始卷积层
    x = layers.Conv2D(64, (3, 3), padding='same', strides=1)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 堆叠残差块
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)

    # 全局平均池化层
    x = layers.GlobalAveragePooling2D()(x)

    # 全连接层 + Softmax
    x = layers.Dense(num_classes, activation='softmax')(x)

    # 创建模型
    model = models.Model(inputs, x, name='ResNet18')
    return model

# 训练和评估模型
def train_and_evaluate_model_resnet():
    # 加载数据
    x_train, y_train, x_test, y_test = load_data()

    # 创建模型
    model = create_resnet_model(input_shape=(32, 32, 3), num_classes=10)  # CIFAR-10 图像大小是 32x32

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

    # 评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')

#2、inception
# Inception Block：包含四个分支（1x1卷积，3x3卷积，5x5卷积，最大池化）
def inception_block(x, filters):
    # 1x1 卷积
    branch1x1 = layers.Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)

    # 1x1 卷积 -> 3x3 卷积
    branch3x3 = layers.Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
    branch3x3 = layers.Conv2D(filters[2], (3, 3), padding='same', activation='relu')(branch3x3)

    # 1x1 卷积 -> 5x5 卷积
    branch5x5 = layers.Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    branch5x5 = layers.Conv2D(filters[4], (5, 5), padding='same', activation='relu')(branch5x5)

    # 3x3 最大池化 -> 1x1 卷积
    branch_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = layers.Conv2D(filters[5], (1, 1), padding='same', activation='relu')(branch_pool)

    # 拼接所有分支的输出
    output = layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1)
    return output


# 定义 Inception 网络
def create_inception_model(input_shape=(224, 224, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # 初始卷积层
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Inception Block
    x = inception_block(x, [64, 128, 128, 32, 32, 32])  # Example filter sizes for each branch

    # 全局平均池化层
    x = layers.GlobalAveragePooling2D()(x)

    # 全连接层 + Softmax 输出层
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, x, name='InceptionV1')
    return model

# 训练和评估模型
def train_and_evaluate_model_inception():
    # 加载数据
    x_train, y_train, x_test, y_test = load_data()

    # 创建模型
    model = create_inception_model(input_shape=(32, 32, 3), num_classes=10)  # CIFAR-10 图像大小为 32x32

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

    # 评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')


#3、mobilenet
# 深度可分离卷积层
def depthwise_separable_conv(x, filters, kernel_size=3, strides=1, padding='same'):
    # 深度卷积（Depthwise Convolution）
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding=padding, activation='relu')(x)
    # 逐点卷积（Pointwise Convolution）
    x = layers.Conv2D(filters, (1, 1), padding='same', activation='relu')(x)
    return x


# 构建 MobileNetV1 网络
def create_mobilenet_model(input_shape=(224, 224, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # 初始卷积层
    x = layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu')(inputs)

    # 深度可分离卷积层（使用多个深度可分离卷积块）
    x = depthwise_separable_conv(x, 64)
    x = depthwise_separable_conv(x, 128, strides=2)
    x = depthwise_separable_conv(x, 128)
    x = depthwise_separable_conv(x, 256, strides=2)
    x = depthwise_separable_conv(x, 256)
    x = depthwise_separable_conv(x, 512, strides=2)

    # 使用更多的深度可分离卷积层
    for _ in range(5):
        x = depthwise_separable_conv(x, 512)

    # 全局平均池化
    x = layers.GlobalAveragePooling2D()(x)

    # 全连接层 + Softmax 输出层
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, x, name='MobileNetV1')
    return model

# 训练和评估模型
def train_and_evaluate_model_mobilenet():
    # 加载数据
    x_train, y_train, x_test, y_test = load_data()

    # 创建模型
    model = create_mobilenet_model(input_shape=(32, 32, 3), num_classes=10)  # CIFAR-10 图像大小为 32x32

    # 编译模型
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 训练模型
    model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

    # 评估模型
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')


# 加载 CIFAR-10 数据集
def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # 将像素值归一化到 [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 将标签转换为 one-hot 编码
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

# 运行训练和评估
if __name__ == "__main__":
    train_and_evaluate_model_resnet()       #作业1
    train_and_evaluate_model_inception()        #作业2
    train_and_evaluate_model_mobilenet()        #作业3



