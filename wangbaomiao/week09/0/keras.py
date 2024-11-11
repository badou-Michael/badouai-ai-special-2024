# -*- coding: utf-8 -*-
# time: 2024/11/8 10:18
# file: keras.py.py
# author: flame
'''
将训练数据集加载到内存中
'''
from tensorflow.keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
print("train_images.shape : ",train_images.shape)
print("train_label : ",train_labels)
print("test_images.shape : ",test_images.shape)
print("test_label : ",test_labels)
''' 打印结果如下:
train_images.shape :  (60000, 28, 28)
train_label :  [5 0 4 ... 5 6 8]
test_images.shape :  (10000, 28, 28)
test_label :  [7 2 1 ... 4 5 6]
'''
'''
打印出用于测试的第一张图片出来看看
'''
digit = test_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit,cmap = plt.cm.binary)
plt.show()

'''
使用tensorflow搭建一个有效识别的神经网络
1 layers：表示神经网络的一个有效处理层，dense:全链接曾
2 models.Sequential()：表示把每一个数据处理层连接起来
3 layers.Dense() ：构造一个数据处理层
4 input_shape(28*28,) ：表示当前数据处理层接收的数据格式必须是长和宽28*28的二维数组
后面的","表示数组里面的每一个元素到底包含多少个数字都没有关系
'''
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer='rmsprop',loss='categorical_crissentropy',metrics=['accuracy'])

'''
将数据输入到网络模型之前 先对数据做归一化处理
1 reshaoe(60000,28,28) train_images,有600000个元素，每个元素是28行28列的二维数组，现在需要将他转为一维数组
2 由于数字图案是一个灰度图，图片中每个像素点的值转为0-255之间
3 train_images.astype("float32") / 255 把每个像素点的值从0-255 转为0-1之间
'''
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype("float32") / 255

'''
把图片上的标记也做一个更改：
目前图片上的数字图案是0-9
列入test_images[0]对于的数字7 的手写图案，那么其对应的标记test_labels[0]的值就是7
我们需要把7变成一个10个元素的数组，数组中从0开始第8个元素设置为1，其他设置成0表示对应的数字是7
例如test_label[0]的值由7转变为数组[0,0,0,0,0,0,0,1,0,0] -- one hot
'''
from tensorflow.keras.utils import to_categorical

print("before change : ", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change : ", test_labels[0])

'''
将数据输入神经网络进行训练：
train_images : 用于训练的手写图片
train_labels : 对应的图片的标记
batch_size : 每次从网络输入的数组中随机选择128个作为一组进行计算
epochs : 每次计算的循环是多少次，代，一代经过的是完整的数据集
'''
network.fit(train_images,train_labels,epochs=5,batch_size=128)

'''
测试数据输入，检验神经网络学习后的识别效果
识别效果与硬件有关( GPU/CPU )
'''
test_loss,test_acc = network.evaluate(test_images,test_labels,verbose=1)
print("test_loss : ",test_loss)
print("test_acc : ",test_acc)

'''
输入一张手写图片到神经网络中，验证它的效果
'''
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((600000,28*28))
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("图片中的数字是 ： ",i)
        break