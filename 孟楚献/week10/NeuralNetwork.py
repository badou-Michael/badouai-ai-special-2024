from unittest.mock import right

import matplotlib.pyplot as plt
import numpy as np
import numpy.random
import scipy.special

# 简单神经网络
# 1.初始化网络接口，网络层结构，学习率等参数
# 2.训练接口，用训练数据更新链路权重
# 3.推理接口
class NeuralNetWork:
    # 三个层，输入层，中间层，输出层，学习率
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 网络结构
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        # 学习率
        self.learning_rate = learning_rate
        # 初始化权重参数
        self.wih = numpy.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        self.who = numpy.random.rand(self.output_nodes, self.hidden_nodes) - 0.5
        # 初始化激活函数
        self.activation = lambda x : scipy.special.expit(x)

    def query(self, inputs):
        # 计算中间层输入
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算中间层输出
        hidden_outputs = self.activation(hidden_inputs)
        # 计算输出层输入
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算最终输出
        final_outputs = self.activation(final_inputs)
        return final_outputs

    # 输入数据列表，正确结果列表
    def train(self, input_list, target_list):
        # 转为二维矩阵 n * shape
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
        # 计算中间层输出
        hidden_outputs = self.activation(numpy.dot(self.wih, inputs))
        # 计算输出层输出
        final_outputs = self.activation(numpy.dot(self.who, hidden_outputs))

        # 计算误差
        output_errors = targets - final_outputs
        # 反向传播
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 计算权重更新量，加到权重上
        self.who += self.learning_rate * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                                    numpy.transpose(hidden_outputs))
        self.wih += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                                    numpy.transpose(inputs))


data_file = open("./NeuralNetWork_从零开始/dataset/mnist_test.csv")
data_list = data_file.readlines()
data_file.close()
print(data_list)
print(len(data_list))
all_values = data_list[0].split(',')
print(all_values)
print(len(all_values))
img_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
plt.imshow(img_array, cmap='gray', interpolation='None')
plt.show()
#数据预处理（归一化）
scaled_input = img_array / 255.0 * 0.99 + 0.01
print(scaled_input)

# 网络运行
input_nodes = 28 * 28
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
epoch = 5
network = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# 读取训练数据
train_data_file = open("NeuralNetWork_从零开始/dataset/mnist_train.csv")
train_data_list = train_data_file.readlines()
train_data_file.close()
print(len(train_data_list))
inputs_list = []
target_list = []
for train_data in train_data_list:
    all_values = train_data.split(',')
    inputs_list.append((numpy.asfarray(all_values[1:])) / 255 * 0.99 + 0.01)
    target = np.zeros(output_nodes) + 0.01
    target[int(all_values[0])] = 0.99
    target_list.append(target)

for e in range(epoch):
    for inputs, targets in zip(inputs_list, target_list):
        network.train(inputs, targets)

test_data_file = open("NeuralNetWork_从零开始/dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
inputs_list.clear()
correct_num_list = []
for test_data in test_data_list:
    all_values = test_data.split(',')
    inputs_list.append((numpy.asfarray(all_values[1:])) / 255 * 0.99 + 0.01)
    correct_num_list.append(int(all_values[0]))

total = 0; right_count = 0
for inputs, correct_number in zip(inputs_list, correct_num_list):
    outputs = network.query(inputs)
    label = numpy.argmax(outputs)
    print("正确数字为", correct_number)
    print("神经网络预测的结果为", label)
    total = total + 1
    if correct_number == label:
        right_count = right_count + 1
print("神经网络的正确率为", right_count / total)
