#-*- coding:utf-8 -*-
# author: 王博然
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models,layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape: ', train_images.shape)   # (60000, 28, 28)
print('train_labels.shape: ', train_labels.shape)   # (60000,)
print('first 10 train_labels: ', train_labels[:10]) # [5 0 4 1 9 2 1 3 1 4]
print('test_images.shape: ', test_images.shape)     # (10000, 28, 28)
print('test_labels.shape: ', test_labels.shape)     # (10000,)
print('first 10 test_labels: ', test_labels[:10])   # [7 2 1 0 4 1 4 9 5 9]
# 图片的大小是28*28

# digit = test_images[5]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

# 模型设置及初始化
network = models.Sequential() # 把每一个数据处理层串联起来
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
## 优化器: rmsprop, 损失函数: 交叉熵,
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 数据处理
## train_images数组原来含有60000个元素，每个元素是一个28行，28列的二维数组
## 现在把每个二维数组转变为一个含有28*28个元素的一维数组
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255 # 归一化

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

# one hot
print("before change, train_label: ", train_labels[0], " test_label: ", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change, train_label: ", train_labels[0], " test_label: ", test_labels[0])

# 训练
network.fit(train_images, train_labels, epochs=1, batch_size=128)

# 测试
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print("loss: ",test_loss, ", test_acc: ", test_acc)

# 验证
test_img = test_images[:5]  # 已经reshape过的值
res = network.predict(test_img)  # test_img: (5, 784)
for i in range(5): # 得到的res (5, 10)
    print("pic no%d is %d, probability: %f." % (i + 1, np.argmax(res[i]), np.max(res[i])))