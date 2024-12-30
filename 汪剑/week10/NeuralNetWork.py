import numpy
import matplotlib.pyplot as plt
import scipy.special


class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化网络，设置输入层，中间层和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 设置学习率
        self.lr = learningrate

        '''
        初始化权重：
        输入层 → 中间层：wih
        中间层 → 输出层：who
        '''
        # self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        # self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        '''
        scipy.special.expit 对应 sigmod函数
        '''
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        # 根据输入的训练数据更新节点链路权重
        '''
        把 inputs_list,targets_list 转换成 numpy 支持的二维矩阵
        .T 表示做矩阵转置
        '''
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 计算信号经过输入层后产生的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)

        # 中间层神经元对输入的信号做激活函数后得到的输出信号
        hidden_outputs = self.activation_function(hidden_inputs)

        # 输出层接收来自中间层的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # 输出层对信号量进行激活函数后得到最终的输出信号
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        outputs_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, outputs_errors * final_outputs * (1 - final_outputs))

        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * numpy.dot((outputs_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    def query(self, inputs):
        # 计算从输入层到中间层传递的信号量：WX + b
        hidden_inouts = numpy.dot(self.wih, inputs)

        # 计算中间层经过激活函数传递的信号量：sigmod
        hidden_outputs = self.activation_function(hidden_inouts)

        # 计算中间层到输出层传递的信号量：WX + b
        final_inputs = numpy.dot(self.who, hidden_outputs)

        # 计算输出层神经元经过激活函数输出的信号量：sigmod
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs


# 初始化网络
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读取训练数据
# open函数里的路径根据数据存储的路径来设定
training_data_file = open('../dataset/mnist_train.csv')
training_data_list = training_data_file.readlines()
training_data_file.close()

print(training_data_list)

epochs = 5

for e in range(epochs):
    # 把数据依靠 ',' 区分，并分别读入
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

# 读取推理数据
test_data_file = open('../dataset/mnist_test.csv')
test_data_list = test_data_file.readlines()
test_data_file.close()

'''
最后我们把所有的测试图片都输入网络，看看它检测的效果如何
'''
scores = []
for record in test_data_list:
    all_values = record.split(',')
    collect_number = int(all_values[0])
    print('该图片对应的数字为:', collect_number)

    # 预处理数字图片
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01

    # 让网路判断图片对应的数字，推理
    outputs = n.query(inputs)
    # 找到数值最大的神经元对应的编号
    label = numpy.argmax(outputs)
    print('output reslut is : ', label)
    if label == collect_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

# 计算图片判断的成功率
scores_array = numpy.asarray(scores)  # numpy.asarray 将输入转换为数组
print('perfermance = ', scores_array.sum() / scores_array.size)
