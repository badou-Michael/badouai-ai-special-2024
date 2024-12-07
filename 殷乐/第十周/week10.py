import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
import random

'''
########################################################## 1.从零实现训练过程 ##########################################################
class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化网络，设置输入层，中间层，和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 学习率
        self.lr = learningrate

        # 随机输入权重
        self.wih = (np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        # sigmoid作为激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        # 根据输入的训练数据更新节点链路权重

        # 把inputs_list, targets_list转换成numpy支持的二维矩阵
        #.T表示做矩阵的转置

        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # 计算信号经过输入层后产生的信号量
        hidden_inputs = np.dot(self.wih, inputs)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层接收来自中间层的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        # 输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        np.transpose(inputs))

        pass

    def query(self, inputs):
        # 根据输入数据计算并输出答案
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最外层接收到的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# 初始化网络
# 输入图片 像素28*28
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读入训练数据
# open函数里的路径根据数据存储的路径来设定
training_data_file = open("dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 加入epochs,设定网络的训练循环次数
epochs = 5
for e in range(epochs):
    # 把数据依靠','区分，并分别读入
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        # 设置图片与数值的对应关系
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    # 预处理数字图片
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 让网络判断图片对应的数字
    outputs = n.query(inputs)
    # 找到数值最大的神经元对应的编号
    label = np.argmax(outputs)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)

# 计算图片判断的成功率
scores_array = np.asarray(scores)
print("performance = ", scores_array.sum() / scores_array.size)
'''

'''
# 2.使用tf实现神经网络训练和推理
import tensorflow as tf
# 使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))  # 加入偏置项
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)  # 加入激活函数

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))  # 加入偏置项
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)  # 加入激活函数

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y - prediction))
# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# tf.Session() 启动图，with关闭图，防止内存泄漏，否则要用close()
with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 训练2000次,run是跑图
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
    plt.show()
'''

# 3.使用pytorch实现手写数字识别
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])

    trainset = torchvision.datasets.MNIST(root='./dataset', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./dataset', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
    return trainloader, testloader


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


net = MnistNet()
model = Model(net, 'CROSS_ENTROPY', 'RMSP')
train_loader, test_loader = mnist_load_data()
model.train(train_loader)
model.evaluate(test_loader)