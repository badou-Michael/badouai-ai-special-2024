# -*- coding: utf-8 -*-
# time: 2024/11/8 18:12
# file: NeturalNetWork.py
# author: flame
import numpy
import scipy.special


class NuturalNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = (numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes)))
        self.who = (numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes)))
        self.activation_function = lambda x:scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        # 计算输出信号量
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        # 计算输入层到中间层的信号量
        hidden_inputs = numpy.dot(self.wih,inputs)
        # 计算中间层经过激活函数后的信号量
        hidden_outpus = self.activation_function(hidden_inputs)
        # 计算中间层到输出层的信号量
        final_inputs = numpy.dot(self.who,hidden_outpus)
        # 计算输出层经过激活函数后的信号量
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        outputs_errors = targets - final_outputs
        '''
        在正向传播时，numpy.dot(self.wih,inputs)，numpy.dot(self.who,hidden_outpus)都是直接使用矩阵，因此他们维度是确定的匹配的
        在反向传播时，numpy.dot(self.who.T,需要转置self.who以确保误差传播的矩阵乘法维度相匹配'''
        hidden_errors = numpy.dot(self.who.T,outputs_errors * final_outputs * (1 - final_outputs))
        #  根据误差更新权重
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outpus * (1 - hidden_outpus)),numpy.transpose(inputs))
        self.who += self.lr * numpy.dot((outputs_errors * final_outputs * (1 - final_outputs)),numpy.transpose(hidden_outpus))
        pass

    def query(self, inputs_list):
        # 计算输入层到中间层的信号量
        hidden_inputs = numpy.dot(self.wih,inputs_list)
        # 计算中间层经过激活函数后的信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算中间层到输出层的信号量
        final_inputs = numpy.dot(self.who,hidden_outputs)
        # 计算输出层经过激活函数后的信号量
        final_outpus = self.activation_function(final_inputs)
        print(final_outpus)
        return final_outpus

# 初始化神经网络
'''
一张图片共有28*28 784个数值 因此神经网络的输入节点需要具备784个输入节点'''
input_nodes = 784
hidden_node = 200
output_nodes = 10
learning_rate = 0.1
n = NuturalNetwork(input_nodes,hidden_node,output_nodes,learning_rate)

# 读入训练数据
training_data_file = open("../dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 加入 epochs，设定神经网络的训练循环次数
epochs = 50
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)

test_data_file = open("../dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("当前图片对应的数字是 ： ",correct_number)
    # 处理输入数据
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    outputs = n.query(inputs)
    # 找到数值最大的神经元对应的编号
    label = numpy.argmax(outputs)
    print("预测结果是：",label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print("正确率：",scores)

# 计算图片判断的成功率
scores_array = numpy.asarray(scores)
print("正确率：",scores_array.sum()/scores_array.size)