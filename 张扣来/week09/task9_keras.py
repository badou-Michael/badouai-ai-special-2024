from tabnanny import verbose

from tensorflow.keras.datasets import mnist

'''
mnist：这是Keras中的一个模块，提供了MNIST数据集的加载功能。
load_data()：这是mnist模块中的一个函数，用于加载MNIST数据集。
train_images是用于训练系统的手写数字图片;
train_labels是用于标注图片的信息;
test_images是用于检测系统训练效果的图片；
test_labels是test_images图片对应的数字标签
'''
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
print('train_images.shape:\n',train_images.shape)
print('train_labels:\n',train_labels)
print('test_images.shape:\n',test_images.shape)
print('test_labels:\n',test_labels)

# # 打印第一张用于测试的图
# digit = test_images[0]
# import matplotlib.pyplot as plt
# # cmap=plt.cm.binary指定了颜色映射，binary表示黑白颜色映射
# plt.imshow(digit,cmap = plt.cm.binary)
# plt.show()

# 创建模型和添加层
from tensorflow.keras import models
from tensorflow.keras import layers
# 创建一个空模型，命名为network。
network = models.Sequential()
# 模型中添加一个全连接层（Dense layer），这个层有512个神经元，使用ReLU激活函数。
# input_shape=(28*28)指定了输入数据的形状，即每个输入样本有28*28=784个特征
network.add(layers.Dense(512,activation = 'relu',input_shape = (28*28,)))
# 第二层 有10个神经元，使用softmax激活函数
network.add(layers.Dense(10,activation='softmax'))
# 编译模型，指定优化器为rmsprop，损失函数为categorical_crossentropy，
# 指定评估模型性能的指标为准确率（accuracy）。
network.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy',metrics = ['accuracy'])

# 将训练图像数据从原始的三维形状（60000, 28, 28）重塑为二维形状（60000, 784）。
# 这是因为在神经网络中，输入数据通常需要是一维或二维的，而784是28乘以28的结果，即每个图像的像素总数。
train_images = train_images.reshape((60000,28 *28))
print('train_images二维化后数据：\n',train_images)
# 归一化处理
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') / 255

# 使用 to_categorical 函数来将标签转换为 one-hot 编码格式
'''
One-hot 编码是一种常用的数值表示方法，特别是在处理分类数据时。
它将分类变量的每个类别表示为一个二进制向量，除了表示该类别的一个位置为1之外，其余位置都是0。
这种表示方法在机器学习和深度学习中非常有用，因为它可以方便地用于模型训练，特别是当使用基于梯度的优化算法时。
特点
--稀疏性：One-hot 编码使得数据表示非常稀疏，大部分元素都是0，只有少数几个元素是1。
--无序性：One-hot 编码不包含任何关于类别之间顺序的信息，这在处理名义变量（Nominal Variables）时是合适的。
--易于处理：对于机器学习算法来说，特别是线性模型，one-hot 编码使得计算变得更加简单和直观。
'''
from tensorflow.keras.utils import to_categorical
print('before change:\n',test_labels[15])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('after change:\n',test_labels[15])

# 把数据输入模型网络进行训练
'''
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次
'''
network.fit(train_images,train_labels,epochs=5,batch_size = 128)

# 测试数据输入，检验网络学习后的图片识别效果，识别效果与硬件有关
# 评估一个已经训练好的神经网络模型在测试集上的性能。
# 这里使用的是Keras中的evaluate方法，它计算并返回模型在给定数据上的损失值和准确率。
test_loss,test_acc = network.evaluate(test_images,test_labels,verbose=1)
print('test_loss:\n',test_loss)
print('test_acc:\n',test_acc)

# 借用数据集里的照片去测试，也可以手写数字照片，看看效果
import matplotlib.pyplot as plt
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
digit = test_images[2]
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000,28*28))
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] ==1):
        print('这张照片的数字是：\n',i)
        break









