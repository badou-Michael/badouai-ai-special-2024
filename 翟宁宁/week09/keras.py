'''
使用keras框架实现 手写数字  0-9 识别
1. 加载keras内置的 数据集  （60000,28,28） 60000万张28*28手写图像数据
2. 用keras构建网络模型

3. 输入数据标准化
'''
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)  #训练数据集 60000
print('tran_labels = ', train_labels)  #样本标签  0 - 9
#训练数据集 10000
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)


# #随意输出一张图片
# digit = test_images[2]
# import matplotlib.pyplot as plt
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()



#2
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()  #创建一个空神经元模型
#输入层28*28=784神经元
#隐藏层 512神经元
#输出层 10神经元（softmax 方式表示 ，数组对应下标表示数字）
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

#loss 交叉熵
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])


train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255



'''
把图片对应的标记也做一个更改：
目前所有图片的数字图案对应的是0到9。
例如test_images[0]对应的是数字7的手写图案，那么其对应的标记test_labels[0]的值就是7。
我们需要把数值7变成一个含有10个元素的数组，然后在第8个元素设置为1，其他元素设置为0。
例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0] ---one hot
'''
from tensorflow.keras.utils import to_categorical
print("before change:" ,test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])


'''
把数据输入网络进行训练：
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次
'''
network.fit(train_images, train_labels, epochs=5, batch_size = 128)


'''
测试数据输入，检验网络学习后的图片识别效果.
识别效果与硬件有关（CPU/GPU）.
'''
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)


'''
输入一张手写数字图片到网络中，看看它的识别效果
'''
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break
