[1]
'''
将训练数据和检测数据加载到内存中：
train_images 是用于训练系统的手写数据图片
train_labels 是用于标注图片的信息
test_images 是用于检测系统训练效果的图片
test_labels 是test_images图片对应的数字标签
'''

'''
两种方式使用keras：
1. 从tensorflow引入
2. 单独下载keras再导入
'''
from tensorflow.keras.datasets import mnist
import numpy as np
import cv2
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ', train_images.shape)
print('train_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels = ', test_labels)

'''
1. train_images.shape 打印结果表明，train_images是一个含有60000个元素的数组
数组中的元素是一个二维数组，行列均为28，即一张图片大小是 28*28

2. train_labels 打印的结果表明，第一张手写数字图片的内容是5，第二张图片是0，以此类推

3. test_images.shape 打印结果表示，用于检测训练效果的图片有10000张

4. test_labels 输出结果表明，用于检测的第一张图片内容是数字7，第二张是数字2，以此类推
'''

[2]
'''
把用于测试的第一张图片打印出来看看
'''
digit = test_images[0]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

[3]
'''
使用TensorFlow.Keras搭建一个有效识别图案的神经网络
1. layers：表示神经网络中的一个数据处理层（dense：全连接层）
2. models.Sequential()：表示把每一个数据处理层串联起来
3. layers.Dense(...)：构造一个数据处理层
4. input_shape(28*28)：表示当前处理层接收的数据格式必须是长和宽都是28的二维数组
后面的 “,” 表示数组里面的每一个元素到底包含多少个数字都没关系
'''
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()  # 声明一个空的模型（定义一个串联网络&一个容器）

'''
Dense()  就是 WX + b 的接口，全连接

512 就是隐藏层有512个节点
激活函数是 relu
输入节点个数是 28*28，相当于每一个像素点都作为输入
'''
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))

'''
输出节点个数是 10 （输出节点个数 = 类别个数，0~9共10个数字）
'''
network.add(layers.Dense(10, activation='softmax'))

# 优化项
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

[4]
'''
在把数据输入到网络模型之前，把数据做归一化处理：
1. reshape(60000,28*28)：train_images数组原来含有60000个元素，每一个元素是一个28行，28列的二维数组，
现在把每个二维数组转变为一个含有28*28个元素的一维数组
2. 由于数字图案是一个灰度图，图片中每个像素点值的大小范围在0到255之间
3. train_images.astype('float32')/255 把每个像素点的值从范围0-255转变为范围在0-1之间的浮点值
'''
train_images = train_images.reshape(60000, 28 * 28)
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape(10000, 28 * 28)
test_images = test_images.astype('float32') / 255

'''
把图片对应的标记也做一个更改：
目前所有图片的数字图案对应的是0到9
例如test_images[0] 对应的数字是7的手写图案，那么其对应的标记test_labels[0]的值就是7
我们需要把数值7变成一个含有10个元素的数组，然后在第8个元素设置1，其余元素设置为0
例如test_labels[0]的值由7转变为数组 [0,0,0,0,0,0,0,1,0,0]  --one hot
'''
from tensorflow.keras.utils import to_categorical

print('before change:', test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('after change:', test_labels[0])

[5]
'''
把数据输入网络进行训练：
train_images：用于训练的手写数字图片
train_labels：对应的是图片的标记
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算
epochs：每次计算的循环是5次
'''
network.fit(train_images, train_labels, epochs=5, batch_size=128)

[6]
'''
测试数据输入，检验网络学习后的图片识别效果
识别效果与硬件有关（CPU/GPU）
'''
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)  # verbose 是否打印日志
print(test_loss)
print('test_acc:', test_acc)

# test_images = test_images.reshape(10000, 28 * 28)
# res = network.predict(test_images)
#
# print(res[1])
#
# for i in range(res[1].shape[0]):
#     if (res[1][i] == 1):
#         print('the number for the picture is :', i)
#         break

img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_CUBIC)
# img = img.astype('float32') / 255
# print(img)

res = network.predict(img.reshape(1, 28 * 28))  # network.predict(batch_size,features)

for i in range(res[0].shape[0]):
    if (res[0][i] == 1):
        print('the number for the picture is :', i)
    else:
        print('Not found')
        break
