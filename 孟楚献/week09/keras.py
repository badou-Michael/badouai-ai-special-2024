from os import system

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from tensorflow.contrib.model_pruning import train
from tensorflow.keras.datasets import mnist

# 1.拿到数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images[0])
print(train_labels)


# 2，创建一个窗口，显示图片
plt.imshow(train_images[3], cmap=plt.cm.binary)
plt.show()

# 3.搭建神经网络
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])

# 4.对数据归一化处理
train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])
test_images = test_images.astype('float32') / 255
print(train_images.shape)
#   更改label格式
from tensorflow.keras.utils import to_categorical
print(train_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(train_labels[0])

# 5.输入数据训练网络
network.fit(train_images, train_labels, 64, 2, 2, )

# 6.测试集检验网络学习成果
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print("test loss:", test_loss)
print("test acc:", test_acc)

# 7.用一下这个网络
img = np.zeros((28, 28))
for i in range(10, 16):
    img[:, i] = 1
for j in range(0, 6), range(24, 28):
    img[j, :] = 0
plt.imshow(img, cmap=plt.cm.binary)
plt.show()
img = img.reshape((1, 28 * 28))
res = network.predict(img)
print(res)
for i in range(res[0].shape[0]):
    if (res[0][i] == 1):
        print("the number for the picture is : ", i)
        break
