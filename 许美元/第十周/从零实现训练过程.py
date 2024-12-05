

import numpy
import scipy.special


class NeuralNetWork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        '''初始化网络，设置 输入层、中间层、输出层 的节点数'''
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.outodes = output_nodes
        # 设置学习率
        self.lr = learning_rate
        '''
        初始化权重矩阵。
        由于权重不一定都是正的，它完全可以是负数，因此我们在初始化时，把所有权重初始化为-0.5到0.5之间
        我们有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
        一个是who,表示中间层和输出层间链路权重形成的矩阵
        '''
        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = numpy.random.rand(self.outodes, self.hnodes) - 0.5
        '''
        scipy.special.expit对应的是sigmod函数.
        lambda是Python关键字，类似C语言中的宏定义.
        我们调用self.activation_function(x)时，编译器会把其转换为spicy.special_expit(x)。
        '''
        self.activation_function = lambda x:scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        '''根据输入的训练数据，更新节点链路权重

        inputs_list:输入的训练数据;
        targets_list:训练数据对应的正确结果。

        训练过程分两步：
        第一步是计算输入训练数据，给出网络的计算结果，这点跟我们前面实现的query()功能很像。
        第二步是将计算结果与正确结果相比对，获取误差，采用误差反向传播法更新网络里的每条链路权重。
        '''
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 计算中间层从输入层接收到的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算中间层经过激活函数后，形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最外层接收到的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算最外层神经元经过激活函数后，输出的信号量
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1-final_outputs))
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))

    def query(self, inputs):
        '''根据输入数据，计算并输出答案'''
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算中间层经过激活函数后，形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最外层接收到的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算最外层神经元经过激活函数后，输出的信号量
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs


input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("dataset/mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        targets = numpy.zeros(output_nodes)
        targets[int(all_values[0])] = 1
        n.train(inputs, targets)

test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:",correct_number)
    #预处理数字图片
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    print("网络认为图片的数字是：", label, '\n----------------\n')
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
#计算图片判断的成功率
scores_array = numpy.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)