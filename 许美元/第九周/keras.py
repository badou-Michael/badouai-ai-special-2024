

'''
将训练数据和检测数据加载到内存中(第一次运行需要下载数据，会比较慢):
train_images是用于训练系统的手写数字图片;
train_labels是用于标注图片的信息;
test_images是用于检测系统训练效果的图片；
test_labels是test_images图片对应的数字标签。

1.train_images.shape打印结果表明，train_images是一个含有60000个元素的数组.
数组中的元素是一个二维数组，二维数组的行和列都是28.
也就是说，一个数字图片的大小是28*28.
2.train_lables打印结果表明，第一张手写数字图片的内容是数字5，第二种图片是数字0，以此类推.
3.test_images.shape的打印结果表示，用于检验训练效果的图片有10000张。
4.test_labels输出结果表明，用于检测的第一张图片内容是数字7，第二张是数字2，依次类推。
'''

from tensorflow.keras.datasets import mnist


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ', train_images.shape)  # (60000, 28, 28)
print('tran_labels = ', train_labels)   # [5 0 4 ... 5 6 8]
print('test_images.shape = ', test_images.shape)  # (10000, 28, 28)
print('test_labels', test_labels)  # [7 2 1 ... 4 5 6]




'''
使用 TensorFlow 的 Keras API 构建了一个简单的神经网络模型，用于处理图像数据（如手写数字识别）。
'''
from tensorflow.keras import models
from tensorflow.keras import layers


network = models.Sequential()
# 创建了一个顺序模型（Sequential），这是一个线性堆叠的层的模型。
# 在顺序模型中，层是按顺序添加的，数据通过前一层流向后一层。
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
# 向模型中添加了第一个全连接层（Dense），它有 512 个神经元。activation='relu'
# 指定了激活函数为 ReLU（Rectified Linear Unit），这是一种常用的激活函数，用于引入非线性特性。
# input_shape=(28*28,) 指定了输入数据的形状，这里假设输入是 28x28 像素的图像，被展平成一维数组（即 784 个特征）。
network.add(layers.Dense(10, activation='softmax'))
# 向模型中添加了第二个全连接层，它有 10 个神经元，对应于 10 个类别（例如，MNIST 数据集中的数字 0-9）。
# activation='softmax' 指定了激活函数为 Softmax，这在多分类问题中常用，用于将输出转换为概率分布。

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# 编译模型，准备训练。
# optimizer='rmsprop' 指定了优化器为 RMSprop（Root Mean Square Propagation），这是一种基于梯度下降的优化算法。
# loss='categorical_crossentropy' 指定了损失函数为分类交叉熵，适用于多分类问题。
# metrics=['accuracy'] 指定了在训练过程中要监控的指标，这里是准确率。


'''
在把数据输入到网络模型之前，把数据做归一化处理:
1.reshape(60000, 28*28）:train_images数组原来含有60000个元素，每个元素是一个28行，28列的二维数组，
现在把每个二维数组转变为一个含有28*28个元素的一维数组.
2.由于数字图案是一个灰度图，图片中每个像素点值的大小范围在0到255之间.
3.train_images.astype(“float32”)/255 把每个像素点的值从范围0-255转变为范围在0-1之间的浮点值。
'''
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255


from tensorflow.keras.utils import to_categorical
# 这个函数用于将整数标签转换为 categorical 类型，即独热编码（one-hot encoding）格式。
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


network.fit(train_images, train_labels, epochs=5, batch_size=128)
# 启动模型的训练过程，使用 train_images 和 train_labels 作为训练数据，
# 进行 5 个 epochs 的训练，每个批次包含 128 个样本。
# 训练过程中，模型将尝试学习最小化损失函数，并通过 fit 方法的返回值来评估训练的效果。
# fit 方法会执行模型的正向传播和反向传播过程，以调整模型的权重，从而最小化损失函数。


test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
# 评估训练好的神经网络模型在测试数据集上的性能。
# evaluate 方法不会更新模型的权重，它只是计算并返回指定的评估指标。
# verbose=1：这个参数控制评估过程中的详细程度。设置为 1 时，将在控制台上输出每个批次的评估结果。如果设置为 0 或者不指定，评估过程将不会输出任何信息。
# test_loss：这是模型在测试数据集上的损失值。损失是模型预测值与实际标签之间差异的量化度量。较低的损失值通常表示模型在测试数据上的表现较好。
# test_acc：这是模型在测试数据集上的准确率，即模型预测正确的样本数占总样本数的比例。


import matplotlib.pyplot as plt


# # 从测试集中取出一张手写数字图像，显示它，然后使用训练好的神经网络模型对其进行预测。
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
# cmap=plt.cm.binary 参数设置颜色映射为二进制（黑白）模式，这样图像将以黑白形式显示。
plt.show()

# print('before reshape:', test_images.shape)
test_images = test_images.reshape((10000, 28*28))
# 模型是在 (10000, 784) 形状上训练的（其中 784 是 28x28 像素图像展平后的一维长度），
# 需要将 test_images 重塑为 (10000, 784)。
res = network.predict(test_images)
# 使用训练好的模型的 predict 方法对重塑后的测试图像进行预测。predict 方法返回每个样本的预测结果。
print(res[1])

for i in range(res[1].shape[0]):
    if res[1][i] == 1:
        print("the number for the picture is : ", i)
        break
