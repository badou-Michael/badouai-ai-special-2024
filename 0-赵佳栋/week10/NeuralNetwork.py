#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：NeuralNetwork.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/12/05 12:22
'''
import numpy
import scipy.special


class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):  # 参数初始化
        # 输入层节点数
        self.inodes = inputnodes
        # 隐藏层节点数
        self.hnodes = hiddennodes
        # 输出层节点数
        self.onodes = outputnodes
        # 学习率
        self.lr = learningrate
        # 输入层到隐藏层的权重矩阵，使用正态分布初始化，均值为 0，标准差为隐藏层节点数的 -0.5 次幂
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        # 隐藏层到输出层的权重矩阵，使用正态分布初始化，均值为 0，标准差为输出层节点数的 -0.5 次幂
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))
        # 定义激活函数为 expit 函数（sigmoid 函数）
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):  # 训练
        # 数据预处理，将输入列表转换为二维数组并转置
        inputs = numpy.array(inputs_list, ndmin=2).T
        # 将目标列表转换为二维数组并转置
        targets = numpy.array(targets_list, ndmin=2).T
        # 计算隐藏层的输入，即输入层乘以输入层到隐藏层的权重矩阵
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 对隐藏层输入进行激活
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层的输入，即隐藏层输出乘以隐藏层到输出层的权重矩阵
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 对输出层输入进行激活
        final_outputs = self.activation_function(final_inputs)

        # 计算输出层的误差，即目标值减去输出层的输出
        output_errors = targets - final_outputs
        # 计算隐藏层的误差，根据输出误差和激活函数的导数
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 基于梯度下降更新隐藏层到输出层的权重矩阵
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                   numpy.transpose(hidden_outputs))
        # 基于梯度下降更新输入层到隐藏层的权重矩阵
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                   numpy.transpose(inputs))
        pass

    def query(self, inputs):  # 推理
        # 计算隐藏层的输入
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 对隐藏层输入进行激活
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层的输入
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 对输出层输入进行激活
        final_outputs = self.activation_function(final_inputs)
        # 打印最终输出结果
        print(final_outputs)
        # 返回最终输出结果
        return final_outputs


# 设置各项参数的初始值
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
# 实例化神经网络模型
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# 读入训练数据
training_data_file = open("dataset/mnist_train.csv", 'r')  # 'r'表示以只读模式打开文件
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 5
for e in range(epochs):
    for record in training_data_list:
        # 用逗号将分类标签值和像素点值分隔开
        all_values = record.split(',')
        # 数据预处理，将像素值归一化到 0.01 到 1.0 之间
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 设置图片标签与输出概率值的对应关系—即进行 one hot 编码
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        # 进行训练
        n.train(inputs, targets)

# 读取测试数据
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

scores = []
for record in test_data_list:
    all_values = record.split(',')
    # 提取测试记录中的正确数字的索引标签
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    # 数据预处理，将像素值归一化到 0.01 到 1.0 之间
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 进行推理
    outputs = n.query(inputs)
    # 找到输出中最大概率的索引作为预测结果
    label = numpy.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

# 计算图片判断的成功率
scores_array = numpy.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)