#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/11/17 17:31
# @Author: Gift
# @File  : Keras-mnist.py
# @IDE   : PyCharm

import keras
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#要结合上面这个导入的mnist才能使用mnist.load_data()
"""
MNIST数据集是由 60000 张训练图像和 10000 张测试图像组成，图像大小是 28 x 28 的灰度图像，已经预先加载在了Keras库中，其中包括4个Numpy数组。
其中 train_images 和 train_labels 组成训练集（training set），模型将从此数据集进行学习，test_images 和 test_labels 一起组成测试集，
用于对模型进行测试。
"""
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(type(train_images), type(train_labels), type(test_images), type(test_labels))
print(train_images.shape, train_labels.shape) # (60000, 28, 28) (60000,)
print(test_images.shape, test_labels.shape) # (10000, 28, 28) (10000,)
print("Test_labels 9997:", test_labels[9997])
print("Test_labels 9998:", test_labels[9998])
print("Test_labels 9999:", test_labels[9999])
# print("第一个元素")
# print(train_images[0])
# print(train_labels[0])
#图像预处理
"""
要对加载的数据进行预处理，以适应网络要求的形状，并将所有值缩放到[ 0, 1] 之间，由于我们训练的图像是 28 x 28 的灰度图，
被保存在 uint8 类型的数组中，也就是值的范围在 [0, 255] 之间，形状为 （60000， 28， 28），所以最后要转换为一个 float32 数组， 
其形状变为 （60000， 28 * 28），取值范围为 0~1
数组重塑之后没有改变元素总的个数和存储方式
"""
train_images = train_images.reshape((60000, 28 * 28))
# print("第一个元素转换后的样子")
# print(train_images[0])
#数据归一化，可不做
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
print("Train images shape:", train_images.shape)
print("Test images shape:", test_images.shape)
#标签编码
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("Train_labels shape:", train_labels.shape)
print("Test_labels shape:", test_labels.shape)

#从测试集上截取三个数据用来自己预测
partial_test_images = test_images[9997:]
partial_test_labels = test_labels[9997:]
test_images = test_images[:9997]
test_labels = test_labels[:9997]
#模型定义
from keras import models, layers
#定义一个空的顺序神经网络
model = models.Sequential()

model.add(layers.Flatten(input_shape=(28*28,)))
#添加一个全连接层
model.add(layers.Dense(512, activation='relu', input_shape=(784,)))
#定义输出层，10种可能
model.add(layers.Dense(10, activation='softmax'))
#编译模型
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#模型训练
train_model = model.fit(train_images,
                        train_labels,
                        epochs=5,
                        batch_size=128,
                         verbose=1)
#模型评估,batch_size默认是32
test_loss, test_acc = model.evaluate(test_images, test_labels,batch_size=200,verbose=1)
print('test_acc:', test_acc)
print("test_loss:", test_loss)
print(train_model.history)
predictions = model.predict(partial_test_images)
print(predictions)
#返回最大值的索引
print(np.argmax(predictions[0]))
print(np.argmax(predictions[1]))
print(np.argmax(predictions[2]))
print(partial_test_labels[0])
print(partial_test_labels[1])
print(partial_test_labels[2])
#可视化
plt.subplot(131)
plt.title("预留数据第一个 ")
plt.imshow(partial_test_images[0].reshape(28, 28))
plt.subplot(132)
plt.title("预留数据第二个 ")
plt.imshow(partial_test_images[1].reshape(28, 28))
plt.subplot(133)
plt.title("预留数据第三个 ")
plt.imshow(partial_test_images[2].reshape(28, 28))
plt.show()
