import os
from tensorflow.keras.datasets import mnist  # 引入了一个默认数据集，是个手写数字的数据集
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
'''
========================================================================================================================
初识keras
'''

# load_data()方法将数据集中的训练图片和标签，测试图片和标签赋值进来
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)

# plt.cm.binary是将图片灰度化的参数，可以实现正常黑白色或者反相的黑白
# plt.imshow(train_images[0], cmap=plt.cm.binary)
# plt.show()


'''
========================================================================================================================
利用keras搭建网络结构
'''

from tensorflow.keras import models
from tensorflow.keras import layers

# 首先利用models实例化一个空的模型
network = models.Sequential()

'''
在layers中调用Dense方法，可以理解为layers是创建层的类，创建一个Dense，Dense是全连接网络
可以理解为，调用了一个layers.Dense()方法，就创建了一个层
这个层是什么层，是输入层还是隐藏层、输出层，取决于它到时候在add时候放在什么位置上，第一个，中间，还是最后一个这种
其实也取决于在Dense中的一些设置参数，毕竟有些层不太一样，只看Dense参数也能大概看出一些区别
'''
# 这就是创建了一个隐藏层
layer1 = layers.Dense(
    512,  # 代表这个隐藏层中有512个神经元节点
    activation='relu',  # 激活函数使用ReLU
    input_shape=(28 * 28,)  # 仅当这是模型中的第一层时需要指定，表示输入数据的形状，此时我们输入的是6万张train数据集，每张28*28的大小
)

# 这是一个输出层
layer2 = layers.Dense(
    10,  # 输出的节点个数 = 最终要识别的内容的类别的数量
    activation='softmax',  # 激活函数使用softmax
)

# 构建神经网络，使用add方法将layer逐一的添加
network.add(layer1)  # 第一层，隐藏层
network.add(layer2)  # 第二层，输出层

# 还要有一个所谓的编译阶段，这个编译阶段实际上就是给network这个模型增加一些可调节的选项
network.compile(
    optimizer='adam',  #
    loss='categorical_crossentropy',  # 这个就是损失函数，交叉熵
    metrics=['accuracy']  # 用来计算准确度的方案，使用accuracy
)

'''
简单模型创建完成
接下来需要对训练集进行一定的处理
现在将训练集和测试集的数据，从28行28列的二维数组，变成有28*28个元素的一维数组，然后对应到训练集的6万组，测试集的1万组
'''

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255.0
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255.0

'''
接下来就是将模型输出的结果对应到标签上
此时需要将识别的标签也量化一下，做成one-hot形式
即如果结果是8，则one-hot结果是[0,0,0,0,0,0,0,1,0]，从头到尾对应的是从0到9
而且one-hot数组也对应上了输出的10个结果
'''

from tensorflow.keras.utils import to_categorical
print(train_labels[0])
print(test_labels[0])
train_labels_oh = to_categorical(train_labels)  # 这里就给one-hot化了
test_labels_oh = to_categorical(test_labels)
print(train_labels_oh[0])
print(test_labels_oh[0])


'''
此时可以将数据输入到刚刚创建的network中进行训练了
训练就调用刚刚模型的fit方法即可
其中，train_images是训练用的数据集，train_labels_oh是训练集对应的正确答案，即标签
'''

network.fit(
    train_images,
    train_labels_oh,
    epochs=5,  # 这个就是之前学的大循环5次
    batch_size=128  # 这个就是batchsize，一次iteration读多少个数据去训练，大小主要要根据设备的性能调整，batchsize设置好后，iteration自动就算出来了
)

'''
当做完fit后，模型在解释器中跑完就说明模型做完了
此时，可以用测试集的数据进行一个测试，看看训练集出来的模型用测试集跑出来的loss值和acc值都是多少
verbose=0：静默模式，不输出任何信息。适合你不需要看到任何训练或评估过程中的输出，或者在进行自动化脚本时使用。
verbose=1：显示进度条。这是默认的行为，Keras 会显示一个动态更新的进度条，显示当前的 epoch 进度、batch 进度、损失值和其他指标（如准确率）。进度条会在每个 batch 结束时更新，并在 epoch 结束时显示最终的结果。
verbose=2：每个 epoch 输出一行。在这种模式下，Keras 不会显示进度条，而是在每个 epoch 结束时输出一行包含该 epoch 的损失值和其他指标的信息。这种方式适合你想要看到每个 epoch 的结果，但不想看到详细的进度条。
'''

test_loss, test_acc = network.evaluate(test_images, test_labels_oh, verbose=1)
print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)

'''
此时做完测试后，就可以上推理了
'''

test_images = test_images.reshape((10000, 28 * 28))
# plt.imshow(q, cmap=plt.cm.binary)
# plt.show()
result = network.predict(test_images)


