# 手写网络训练和测试的完整过程

import scipy.special
import numpy


class NeuralNetwork:
    def __init__(self,  inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        # 权重矩阵,范围[-0.5, 0.5),矩阵形状是(hnodes, inodes),(onodes, hnodes)        
        # self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        # self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5

        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))       
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        self.activation_function = lambda x: scipy.special.expit(x)              
        pass

    # 根据输入的训练数据更新节点链路权重
    def train(self, inputs_list, targets_list):
        '''
        inputs_list:输入的训练数据
        targets_list:训练数据对应的正确结果
        '''

        # 数据预处理
        # 把inputs_list, targets_list转换成numpy支持的二维矩阵，.T表示做矩阵的转置
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 正向过程
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算误差（代公式）                                                               
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))

        # 更新权重                                                                          # 符号是+=,右边本身带有正负号
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))

    # 根据输入数据计算并输出答案
    def query(self, inputs):
        # 根据输入数据计算WX(+b)并输出答案
        # 中间层接收到的输入
        hidden_inputs = numpy.dot(self.wih, inputs)

        # 中间层经过激活函数的输出
        hidden_outputs = self.activation_function(hidden_inputs)

        # 最外层/输出层接收到的输入
        final_inputs = numpy.dot(self.who, hidden_outputs) 

        # 最外层输出
        final_outputs = self.activation_function(final_inputs)

        print(final_outputs)
        return final_outputs


input_nodes, hidden_nodes, output_nodes, learning_rate = 28*28, 100, 10, 0.3
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

train_data_file = open("../../dataset/mnist_test.csv")
train_data_list = train_data_file.readlines()
train_data_file.close()

# 训练
epochs = 10
for e in range(epochs):
    #把数据依靠','区分，并分别读入
    for record in train_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        # inputs = (numpy.asfarray(all_values[1:]))

        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

test_data_list = train_data_list
scores = []    # 把每个图片的推理结果记录，正确为1，错误为0
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)

    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    outputs = n.query(inputs)  # 让网络判断图片对应的数字,推理
    label = numpy.argmax(outputs)  # 找到数值最大的神经元对应的 编号                            
    print("output result is : ", label)

    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)

# 计算正确率
scores_array = numpy.asarray(scores)                                                        
print("perfermance = ", scores_array.sum() / scores_array.size)
