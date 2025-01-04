#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：keras.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/11/22 12:17
'''

from tensorflow.keras.datasets import mnist     # mnist:数据集名称
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()    # 加载数据集
print('train_images\n', train_images.shape)
print('train_labels\n', train_labels)
print('test_images\n', test_images.shape)
print('test_labels\n', test_labels)

one_image = test_images[0]
plt.imshow(one_image)
plt.show()

# 构造模型结构
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))      # 全连接
network.add(layers.Dense(10, activation='softmax'))
# 配置训练过程
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])

# 数据预处理
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# 将标签用one hot方法表示——one hot方法可以让softmax处理后的输出概率值与各个类别一一对应上、从而能够判断测试数据的所属类别
print("before change:", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])

# 进行模型训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 进行模型推理
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print('test_loss', test_loss)
print('test_acc', test_acc)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break