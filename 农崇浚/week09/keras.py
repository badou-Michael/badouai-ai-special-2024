from keras.datasets import mnist
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#加载数据集
(train_img, train_label),(test_img, test_label) = mnist.load_data()

from tensorflow.keras import models
from tensorflow.keras import layers

#添加隐藏层
network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

#编译
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#归一化数据集
train_img = train_img.reshape((60000,28*28))
train_img = train_img.astype('float32')/255

test_img = test_img.reshape((10000,28*28))
test_img = test_img.astype('float32')/255


#标签转变
from tensorflow.keras.utils import to_categorical
print('before change:',test_label[0])
train_label = to_categorical(train_label)
test_label = to_categorical(test_label)
print('after change:',test_label[0])

#把训练集输入进网络
network.fit(train_img,train_label,epochs=5,batch_size=128)

#把测试集输入进网络,检验网络学习以后的正确率
test_loss, test_acc = network.evaluate(test_img,test_label,verbose=1)
print(test_loss)
print(test_acc)

#推理
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]

test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
