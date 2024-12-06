[1]
'''
将训练数据和检测数据加载到内存中(第一次运行需要下载数据，会比较慢):
train_images是用于训练系统的手写数字图片;
train_labels是用于标注图片的信息;
test_images是用于检测系统训练效果的图片；
test_labels是test_images图片对应的数字标签。
'''

from tensorflow.keras.datasets import mnist
(train_images,train_labels),(test_images, test_labels)=mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)
'''
1.train_images.shape打印结果表明，train_images是一个含有60000个元素的数组.
数组中的元素是一个二维数组，二维数组的行和列都是28.
也就是说，一个数字图片的大小是28*28.
2.train_lables打印结果表明，第一张手写数字图片的内容是数字5，第二种图片是数字0，以此类推.
3.test_images.shape的打印结果表示，用于检验训练效果的图片有10000张。
4.test_labels输出结果表明，用于检测的第一张图片内容是数字7，第二张是数字2，依次类推。
'''

[2]
'''
把用于测试的第一张图片打印出来看看
'''
digit=test_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
#cmap参数是matplotlib库中imshow()函数的一个选项，用于指定图像的颜色映射（colormap）。
#plt.cm.binary：黑白两色，用于二值化显示。接近0的值显示为黑色，接近1的值显示为白色。
plt.show()

[3]
'''
使用tensorflow.Keras搭建一个有效识别图案的神经网络，
1.layers:表示神经网络中的一个数据处理层。(dense:全连接层)
2.models.Sequential():表示把每一个数据处理层串联起来.;创建了一个序贯模型，意味着层是按顺序添加的
3.layers.Dense(…):构造一个数据处理层。
4.input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
'''
#layers.Dense(512, activation='relu', input_shape=(28*28,))
# 添加了一个全连接层，有512个神经元，使用ReLU激活函数，输入形状是(28*28,)，因为每张图片是28x28像素。
#layers.Dense(10, activation='softmax')添加了另一个全连接层，有10个神经元，使用softmax激活函数，用于输出10个类别的概率。

from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)) )
network.add(layers.Dense(10, activation='softmax'))

# network.add是Keras中用于向序贯模型添加层的正确方法。
# network.append不是Keras中的方法，不能用于Keras模型。

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy']) #这段代码编译模型，指定了优化器、损失函数和评估指标：
# optimizer='rmsprop'：使用RMSprop优化器。
# loss='categorical_crossentropy'：使用分类交叉熵作为损失函数，适用于多分类问题。
# metrics=['accuracy']：评估模型的准确率。
# 在Keras中，network.compile()方法用于配置模型的参数，这些参数定义了模型在训练过程中的学习算法。具体来说，compile方法设置了以下几个关键部分：
# 优化器（Optimizer）：这是模型在训练过程中用于调整参数以最小化损失函数的算法。在这个例子中，使用的是'rmsprop'优化器，它是一种自适应学习率的优化方法，适合于许多不同类型的问题。
# 损失函数（Loss Function）：这是衡量模型预测与实际结果之间差异的函数。模型训练的目标就是最小化这个损失值。在这个例子中，使用的是'categorical_crossentropy'，这是一个常用于多分类问题的损失函数，它衡量的是模型输出的概率分布与真实标签的概率分布之间的差异。
# 评估指标（Metrics）：这些是在训练和测试过程中用来评估模型性能的指标。在['accuracy']，即准确率，它衡量的是模型正确预测的比例。

[4]
'''
在把数据输入到网络模型之前，把数据做归一化处理:
1.reshape(60000, 28*28）:train_images数组原来含有60000个元素，每个元素是一个28行，28列的二维数组，
现在把每个二维数组转变为一个含有28*28个元素的一维数组.
2.由于数字图案是一个灰度图，图片中每个像素点值的大小范围在0到255之间.
3.train_images.astype(“float32”)/255 把每个像素点的值从范围0-255转变为范围在0-1之间的浮点值。
'''
#这段代码将训练集的图片从28x28的二维数组转换为784（28*28）的一维数组，并将其数据类型转换为float32，然后归一化到0-1之间。
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

##to_categorical函数将整数标签转换为one-hot编码形式，例如，标签7将被转换为[0,0,0,0,0,0,0,1,0,0]。
from tensorflow.keras.utils import to_categorical
print('before change ',test_labels[0])
test_labels=to_categorical(test_labels)
print("after change: ", test_labels[0])
train_labels = to_categorical(train_labels)
print("after change train_labels: ", train_labels[0])


[5]
'''
把数据输入网络进行训练：
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次
'''

network.fit(train_images, train_labels, epochs=5,batch_size=128)

[6]
'''
测试数据输入，检验网络学习后的图片识别效果.
识别效果与硬件有关（CPU/GPU）.
'''
test_loss,test_acc= network.evaluate(test_images, test_labels,verbose=1)
print("test_loss",test_loss)
print("test_acc",test_acc)

[7]
'''
输入一张手写数字图片到网络中，看看它的识别效果
'''
(train_images,train_labels),(test_images,test_labels)= mnist.load_data()
digit=test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

test_images=test_images.reshape(10000,28*28)
res=network.predict(test_images) ##推理接口，上真正考场了，暂时用network.predict(test_images)这个数据展示逻辑
# test_images = test_images.reshape(10000, 28*28) 使用了位置参数来指定新形状的维度。
# test_images = test_images.reshape((10000, 28*28)) 使用了一个元组来指定新形状。
# 在NumPy和Keras中，这两种写法都是可接受的，并且会产生相同的结果。通常，使用元组来指定形状是一种更常见和更清晰的写法，因为它明确地表明了维度的顺序。

for i in range(res[1].shape[0]):
    if res[1][i]==1:
        print('为1的索引是，the number for the picture is :',i)
        break
