import cv2
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
import random


# 归一化到[0,1]
def normalization1(x):
    y = (x-min(x))/(max(x)-min(x))
    return y


# 归一化到[-1,1]
def normalization2(x):
    y = (x-np.mean(x))/(max(x)-min(x))
    return y


# 零均值归一化
def z_score(x):
    y = (x-np.mean(x))/np.std(x)
    # y = (x-np.mean(x))/(np.std(x) * np.std(x))
    return y


# 1.实现标准化
def normalization_test():
    arr = np.array([1, 2, 3, 4, 5])
    print(normalization1(arr))
    print(normalization2(arr))
    print(z_score(arr))


# 2.使用keras实现简单神经网络
def keras_test():
    # 1.加载数据集，并分为训练集和验证集，测试从验证集中找一张
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    # print('train_images.shape = ', train_images.shape) 60000
    # print('tran_labels = ', train_labels)
    # print('test_images.shape = ', test_images.shape)  10000
    # print('test_labels', test_labels)
    # 2.数据处理：数组转换和归一化处理
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255
    # 3.标签处理：one hot
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # 3.构建网络：像素作为输入28*28，512个隐藏层，10个输出；使用全连接+激活函数，最后在加一个softmax做归一化
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    # 编译模型
    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # 训练
    network.fit(train_images, train_labels, epochs=5, batch_size=128)
    # 使用验证集测试
    test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)

    # 在验证集中挑选一张查看效果
    num = random.randint(1, 1000)
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    digit = test_images[num]
    plt.imshow(digit, cmap=plt.cm.binary)
    plt.show()
    test_images = test_images.reshape((10000, 28 * 28))
    res = network.predict(test_images)

    for i in range(res[num].shape[0]):
        print(res[num][i])
        if res[num][i] == 1:
            print("the number for the picture is : ", i)
            break


keras_test()
