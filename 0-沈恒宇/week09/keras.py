from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


# 第一步：获取训练和检测 的 数据、标签
"""
train_images是用于训练系统的手写数字的图片
train_labels是用于标注图片的信息
test_images是用于检测系统训练效果的图片
test_labels是test_images是对应的标签
"""
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)


# 第二步：将用于测试的第一张图片打印出来看看
'''
1.train_images.shape打印结果表明，train_images是一个含有60000个元素的数组.
数组中的元素是一个二维数组，二维数组的行和列都是28.
也就是说，一个数字图片的大小是28*28.
2.train_lables打印结果表明，第一张手写数字图片的内容是数字5，第二种图片是数字0，以此类推.
3.test_images.shape的打印结果表示，用于检验训练效果的图片有10000张。
4.test_labels输出结果表明，用于检测的第一张图片内容是数字7，第二张是数字2，依次类推。
'''
digit = test_images[0]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()


# 第三步：使用TensorFlow.Keras搭建一个有效识别图案的神经网络
"""
1.layers:表示神经网络中的一个数据处理层。(dense:全连接层)
2.models.Sequential():表示把每一个数据处理层串联起来.
3.layers.Dense(…):构造一个数据处理层。
4.input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
这段代码是使用Keras库构建一个简单的神经网络模型的示例。Keras是一个高级神经网络API，它可以运行在TensorFlow、CNTK或Theano之上。下面是这段代码的逐行解释：

1. `network = models.Sequential()`：
   这行代码创建了一个Sequential模型。Sequential是Keras中的一种线性堆叠的神经网络模型，这意味着各层是按顺序添加的，并且每一层的输出会成为下一层的输入。

2. `network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))`：


3. `network.add(layers.Dense(10, activation='softmax'))`：
   这行代码向模型中添加了第二个全连接层，这个层有10个神经元，对应于10个类别的输出。
   使用的激活函数是softmax，这是一种常用于多分类问题输出层的激活函数，它将输出转换为概率分布。

4. `network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])`：
   这行代码编译模型，指定了训练过程中使用的优化器、损失函数和评估指标。
   - `optimizer='rmsprop'`：指定了优化器为RMSprop，这是一种自适应学习率的优化算法，常用于深度学习中。
   - `loss='categorical_crossentropy'`：指定了损失函数为categorical crossentropy，
   这是一种用于多分类问题的损失函数，它衡量的是模型预测的概率分布与真实标签的概率分布之间的差异。
   - `metrics=['accuracy']`：指定了在训练过程中要监控的评估指标，这里使用的是准确率（accuracy），即正确分类的样本占总样本的比例。

总结来说，这段代码定义了一个用于多分类问题的简单神经网络，它包含两个全连接层，第一层用于提取特征，第二层用于分类。
模型使用RMSprop优化器和categorical crossentropy损失函数进行训练，并监控准确率指标。
这个模型可能用于处理类似MNIST手写数字识别这样的任务，其中输入是28x28像素的灰度图像，输出是10个类别（0到9的数字）中一个的概率分布。

"""
network = models.Sequential()
"""这行代码创建了一个Sequential模型。
Sequential是Keras中的一种线性堆叠的神经网络模型，这意味着各层是按顺序添加的，并且每一层的输出会成为下一层的输入。"""

network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
"""这行代码向Sequential模型中添加了一个全连接层（Dense层）。这个层有512个神经元，使用的激活函数是ReLU（Rectified Linear Unit）。
   `input_shape=(28*28,)`指定了输入数据的形状，这里`28*28`意味着每个输入样本是一个784维的向量（例如，28x28像素的图像展平后的向量）。"""

network.add(layers.Dense(10,activation='softmax'))
"""这行代码向模型中添加了第二个全连接层，这个层有10个神经元，对应于10个类别的输出。
   使用的激活函数是softmax，这是一种常用于多分类问题输出层的激活函数，它将输出转换为概率分布。"""

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])
"""这行代码编译模型，指定了训练过程中使用的优化器、损失函数和评估指标。
   - `optimizer='rmsprop'`：指定了优化器为RMSprop，这是一种自适应学习率的优化算法，常用于深度学习中。
   - `loss='categorical_crossentropy'`：指定了损失函数为categorical crossentropy，
   这是一种用于多分类问题的损失函数，它衡量的是模型预测的概率分布与真实标签的概率分布之间的差异。
   - `metrics=['accuracy']`：指定了在训练过程中要监控的评估指标，这里使用的是准确率（accuracy），即正确分类的样本占总样本的比例。"""

# 第四步：把数据做归一化处理，把图片对应的标记也做一个更改
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

"""
把图片对应的标记也做一个更改：
目前所有图片的数字图案对应的是0到9。
例如test_images[θ］对应的是数字7的手写图案，那么其对应的标记test_labels[θ］的值就是7。
我们需要把数值7变成一个含有10个元素的数组，然后在第8个元素设置为1，其他元素设置为0。
例如test_labels[θ] 的值由7转变为数组[θ,θ,θ,0,θ,θ,θ,1,θ,θ]---one hot
"""

print("before change:", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change:", test_labels[0])

# 第五步：开始训练
'''
把数据输入网络进行训练：
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次
'''
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 第六步：测试数据输入，检验网络学习后的图片识别效果
"""
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)：

network：这是之前定义的神经网络模型，它已经被编译过，并且已经拟合（训练）了一些数据。
evaluate：这是Keras模型的一个方法，用于评估模型在一个数据集上的性能。它返回模型在测试数据上的表现，通常是损失值和一些指标（如准确率）。
test_images：这是测试数据集的输入部分，包含了用于评估模型性能的图像数据。
test_labels：这是测试数据集的标签部分，包含了与test_images相对应的真实标签。
verbose=1：这是一个参数，用于控制评估过程中的输出信息。verbose=1表示在评估过程中会打印出进度条信息，
            如果设置为0，则不会打印任何信息，如果设置为2，则会打印每个epoch的详细输出。
            
test_loss, test_acc：这是两个变量，用于存储模型评估的结果。
test_loss：存储模型在测试数据集上的损失值，这个值越低表示模型的预测结果与真实标签的差异越小，模型的性能越好。
test_acc：存储模型在测试数据集上的准确率，这个值越高表示模型的预测结果越准确。
"""
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print("test_loss",test_loss)
print('test_acc',test_acc)

# 第七步：输入一张手写数字图片到网络中，看看效果
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000,28*28))
res = network.predict(test_images)
"""res：这是一个数组，其形状取决于模型输出层的配置。
对于分类问题，res将是一个二维数组，其中每一行对应一个输入图像的预测结果，每一列对应一个类别的概率。
对于回归问题，res将是一个一维或二维数组，取决于输出层的配置。

在这段代码中，res是通过调用network.predict(test_images)得到的预测结果。
由于network是一个用于MNIST手写数字识别的分类模型，其输出层有10个神经元，对应于10个数字类别（0到9）。
因此，res是一个形状为(10000, 10)的二维数组，
每一行代表一个测试图像的预测结果，每一列代表一个数字类别的概率。
"""
for i in range(res[1].shape[0]):
    if(res[1][i]==max(res[1])):
        print("The number for the picture is: ",i)
        break
