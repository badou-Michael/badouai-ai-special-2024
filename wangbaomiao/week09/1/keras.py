# -*- coding: utf-8 -*-
# time: 2024/11/21 10:22
# file: keras.py
# author: flame
import numpy as np

[1]
'''
将训练数据和监测数据加载到内存中(第一次运行需要下载数据，会比较慢)'''
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape : ', train_images.shape)
print('train_labels : ', train_labels)
print('test_images.shape : ', test_images.shape)
print('test_labels : ', test_labels)

[2]
digit = test_images[:3]
import matplotlib.pyplot as plt
'''
zip(axes, digits) 将子图数组 axes 和图像数组 digits 按照对应位置组合成一个迭代器。
for ax, digit in zip(axes, digits): 循环遍历每个子图 ax 和对应的图像 digit。
ax.imshow(digit, cmap=plt.cm.binary) 在当前子图 ax 中显示图像 digit，并使用二进制颜色映射 cmap=plt.cm.binary 来显示图像'''
fig, axes = plt.subplots(1,3)
for ax, digit in zip(axes, digit):
    ax.imshow(digit, cmap=plt.cm.binary)
plt.show()

[3]
'''
使用tensorflow.keras 搭建一个有效识别图案的神经网络
1. layers：表示神经网络中的一个数据处理层 Dense:全连接层
2. models.Sequential() : 表示一个线性的模型，层按顺序堆叠，输入数据会从输入层传递到输出层，把每一个数据处理层串联起来
3。layers.Dense() : 构建一个全连接的数据处理层，简称全连接层，参数：unist=128，表示有128个输出层，
activation='relu'：激活函数，激活函数是神经网络中常用的一种函数，它将输入信号转换为输出信号，并激活神经元的输出。
4. input_shape(28*28,)：输入层的形状，这里表示输入的图片大小为28*28，因为是灰度图，所以只有一个通道。
5. softmax : 表示输出层使用softmax激活函数，softmax函数将向量转换为概率分布，每个元素都是非负数，且和为1。
6. compile: 编译模型，参数：optimizer='rmsprop'：优化器，rmsprop是一种常用的优化器，用于训练神经网络。
loss = 'categorical_crossentropy'：损失函数，用于计算模型在训练过程中预测值与真实值之间的差异。
metrics = ['accuracy']：评估指标，用于评估模型的性能，这里使用准确率作为评估指标。'''
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

[4]
'''
在把数据输入到网络模型之前，把数据做归一化处理
1. reshape(60000,28*28) : train_images 数组原来含有60000个元素，每个元素是一个28行，28列的二维数组
现在把每个二维数组转变为一个含有28*28的一维数组
2. 由于数字图案是一个灰度图，图片中的每个像素点的大小范围在0-255之间
RGB 0-255 红绿蓝 RGBA 0-255,0-255 红绿蓝、透明度 
浮点数图 0.0-1.0，归一化的结果，用于深度学习和cnn中 
3. train_images.astype('float32')/255 把每个像素点的值从0-25转变为范围在0-1之间的浮点值'''
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255

'''
把图片对应的标记也做一个更改，目前所有图片对应的数字图案都是0-9
如test_images[0]对应的是数字7的手写图案，那么其对应的标记test_labels[0]的值就是7
我们需要把数值7变成一个含有10个元素的数组，然后在第8个元素设置为1，其他元素设置为0
test_labels[0]的值由7变为[0,0,0,0,0,0,1,0,0] -- one hot
One-Hot 编码是一种将分类数据转换为二进制向量的表示方法。
在这种表示中，每个类别都用一个独立的二进制位表示，只有一个位为1，其余位为0。'''
from tensorflow.keras.utils import to_categorical
print('before change: ', test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('after change: ', test_labels[0])

[5]
'''
把数据输入网络进行训练
train_images: 用于训练的手写数字图片
train_labels: 对应的是图片标记，也就是结果
batch_size: 每次网络从输入的图片数组中随机选取128个作为一组进行计算
epochs： 每次计算的循环是5次 也就是一代'''
network.fit(train_images,train_labels,epochs=5,batch_size=128)

[6]
'''
测试数据的输入，检验网络学习后的图片识别效果
识别效果与硬件有关 cpu/gpu
verbose : 0/1/2/3/4，默认为1，输出识别结果，0为不输出，1为输出识别结果，2为输出识别结果和损失值，3为输出识别结果和准确率，4为输出识别结果、准确率和损失值'''
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print('test_loss ,test_acc : ', test_loss,test_acc)

[7]
'''
输入一张手写数字图片到网络中，验证它的识别效果'''
((train_images, train_labels), (test_images, test_labels)) = mnist.load_data()
import random
random_indics = random.sample(range(len(test_images)),6)
selected_images = [test_images[i] for i in random_indics]

fig, axes = plt.subplots(1, 6, figsize=(15,3))
for ax, image, index in zip(axes, selected_images, random_indics):
    ax.imshow(image,cmap=plt.cm.binary)
    ax.set_title(index)
    ax.axis('off')
plt.show()
test_images = test_images.reshape((10000,28*28))
result = network.predict(test_images)

selected_results = [result[i] for i in random_indics]

for index, result in zip(random_indics, selected_results):
    predicted_class = np.argmax(result)
    print('识别结果为：', index,'result :',predicted_class)