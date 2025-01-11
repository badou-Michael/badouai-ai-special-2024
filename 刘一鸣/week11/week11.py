'''
第十一周作业：
1.实现cifar-10
2.实现alexnet（训练+推理）
3.实现vgg16（推理）
'''

#1、实现cifar-10
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

#加载数据
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# 数据归一化到 [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0
# 将标签展平（从二维变成一维）
y_train = y_train.flatten()
y_test = y_test.flatten()
# 打印数据集信息
print(f"训练集形状: {x_train.shape}, 测试集形状: {x_test.shape}")

#构建模型
def create_cnn_model():
    model = models.Sequential()

    # 第一个卷积层 + 池化层
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    # 第二个卷积层 + 池化层
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # 第三个卷积层
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # 展平层 + 全连接层
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))  # 输出10个类别
    return model

#编译模型
model = create_cnn_model()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 训练模型
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test)
)

# 测试集评估
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"测试集准确率: {test_acc:.2f}")

#2、实现alexnet（训练+推理）
import numpy as np
from tensorflow.keras.datasets import cifar10

#加载和预处理数据
# 加载 CIFAR-10 数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 数据归一化到 [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0
# 展平标签为一维数组
y_train = y_train.flatten()
y_test = y_test.flatten()

#构建 AlexNet 模型
def create_alexnet():
    model = models.Sequential()

    # 第一层卷积：96个卷积核，尺寸11×11，步幅4，输入224×224×3
    model.add(layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # 第二层卷积：256个卷积核，尺寸5×5，步幅1，补零
    model.add(layers.Conv2D(256, (5, 5), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # 第三层卷积：384个卷积核，尺寸3×3
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))

    # 第四层卷积：384个卷积核，尺寸3×3
    model.add(layers.Conv2D(384, (3, 3), padding='same', activation='relu'))

    # 第五层卷积：256个卷积核，尺寸3×3
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))

    # 展平 + 全连接层
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10))  # CIFAR-10 有10个类别

    return model

# 创建模型
alexnet = create_alexnet()

#编译模型
alexnet.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

#训练模型
history = alexnet.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(x_test, y_test)
)

#测试集评估
# 测试集评估
test_loss, test_acc = alexnet.evaluate(x_test, y_test, verbose=2)
print(f"测试集准确率: {test_acc:.2f}")

# 随机从测试集中选取一张图片进行推理
sample_index = np.random.randint(0, len(x_test))
sample_image = x_test[sample_index]
sample_label = y_test[sample_index]

# 进行预测
predictions = alexnet.predict(np.expand_dims(sample_image, axis=0))
predicted_label = np.argmax(predictions)
# 打印预测和真实标签
print(f"真实标签: {sample_label}, 预测标签: {predicted_label}")

#3、实现vgg16（推理）
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

# 加载CIFAR-10数据集
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# 数据预处理：将数据标准化到 [0, 255] 之间
x_test = x_test.astype('float32')

# 加载VGG16预训练模型
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
# 添加分类层（全连接层）
model = models.Sequential()
model.add(vgg_model)  # 将预训练的VGG16模型作为基础
model.add(layers.Flatten())  # 展平层，方便全连接层接入
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout层，防止过拟合
model.add(layers.Dense(10, activation='softmax'))  # CIFAR-10有10个类别
# 冻结VGG16的层，避免训练时更新它的权重
for layer in vgg_model.layers:
    layer.trainable = False

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))

# 推理阶段：使用测试数据进行预测
predictions = model.predict(x_test)

# 获取第一个测试样本的预测结果
predicted_class = np.argmax(predictions[0])
print(f"Predicted class for the first image: {predicted_class}")
