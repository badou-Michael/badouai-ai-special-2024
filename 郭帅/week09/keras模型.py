# keras模型

'''
1、引用接口，加载数据集
train_images是用于训练系统的手写数字图片;
train_labels是用于标注图片的信息;
test_images是用于检测系统训练效果的图片；
test_labels是test_images图片对应的数字标签。
'''
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('train_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels = ', test_labels)


'''
2、在把数据输入到网络模型之前，把数据做归一化处理
reshape((60000, 28*28)) 这一操作将每张图片的二维形状 (28, 28) 转换为一维的形状 (28*28,)，即每张图片变成一个包含 784 个像素值的数组;
astype('float32') 将像素值的数据类型转换为 float32;
/ 255 这一操作将每个像素值从 [0, 255] 范围归一化到 [0, 1] 范围。
'''
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255


'''
3、将分类标签（train_labels 和 test_labels）转换为 one-hot 编码格式
例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0] ---one hot
'''
from tensorflow.keras.utils import to_categorical
# 将标签转换为one-hot编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


'''
4、使用tensorflow.Keras搭建一个有效识别图案的神经网络
layers:表示神经网络中的一个数据处理层。(dense:全连接层)
models.Sequential():表示把每一个数据处理层串联起来
input_shape(28*28,):表示当前处理层接收的数据格式必须是28*28的一维数组
'''
# models 提供了网络模型的结构,layers 提供了实现结构中的各个层
from tensorflow.keras import models, layers
# 创建一个空的神经网络模型
network = models.Sequential()
# 添加第一层：全连接层，512个神经元，激活函数是ReLU,输入(28*28,) 是一个包含一个元素的元组，而 (28*28) 是一个普通的整数,必须带有（,）
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
# 添加第二层：全连接层，10个神经元，激活函数是Softmax
network.add(layers.Dense(10, activation='softmax'))
# 编译模型，指定优化器(自适应学习率优化器，它会在每次参数更新时动态地调整学习率)、损失函数(分类交叉熵)和评估指标(准确率)
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


'''
5、把数据输入网络进行训练：
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次
'''
network.fit(train_images, train_labels, epochs=5, batch_size = 128)


'''
6、测试数据输入，检验网络学习后的图片识别效果.
'''
# test_loss, test_acc是通过调用 network.evaluate(test_images, test_labels) 得到的，是 Keras 内置的标准功能。
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print('test_loss',test_loss)
print('test_acc', test_acc)


'''
7、输入一张手写数字图片到网络中，看看它的识别效果
'''
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_images = test_images.reshape((10000, 28*28))
# 使用训练好的模型进行预测，返回每张图片属于各个数字（0-9）的概率
res = network.predict(test_images)
# res[0] 是对第一张测试图片的预测结果，表示属于 0-9 各个数字的概率
print(res[0])
# 预测出概率最大的类别对应的数字：argmax() 返回数组中最大元素的索引
predicted_class = res[0].argmax()
print(f"Predicted class: {predicted_class}")
