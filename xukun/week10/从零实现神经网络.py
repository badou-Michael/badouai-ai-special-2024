import numpy as np
import scipy.special as sp


class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 设置学习率
        self.lr = learningrate

        '''
        初始化权重矩阵，我们有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
        一个是who,表示中间层和输出层间链路权重形成的矩阵
        '''
        self.wih = (np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        '''
        每个节点执行激活函数，得到的结果将作为信号输出到下一层，我们用sigmoid作为激活函数
        '''
        self.activation_function = lambda x: sp.expit(x)

    def train(self, inputs_list, targets_list):
        # 根据输入的训练数据更新节点链路权重
        '''
        把inputs_list, targets_list转换成numpy支持的二维矩阵
        .T表示做矩阵的转置
        '''
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

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T
        # 输入层到隐藏层
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐藏层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 隐藏层到输出层
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层输出
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.3

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#读取训练数据
training_data_file = open("dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
epochs = 5
# 进行训练
for e in range(epochs):
    # 把数据依靠','区分，并分别读入
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 设置图片与数值的对应关系
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)


#读取测试数据
traning_test_data = open("dataset/mnist_test.csv", 'r')
traning_test_data_lines = traning_test_data.readlines()
traning_test_data.close()

scorecard = []

for data_list in traning_test_data_lines: # 遍历测试数据
    input_list = data_list.split(',')  #读取每一行数据 根据','分割 首位是正确的数字 后面是图片数据
    inputs = np.asfarray(input_list[1:]) / 255.0 * 0.99 + 0.01 # 归一化图片数据
    # 设置图片与数值的对应关系
    target_list = np.zeros(output_nodes) + 0.01
    target_list[int(input_list[0])] = 0.99
    outputNum = n.query(inputs) # 输入图片数据 得到输出结果
    argmax = np.argmax(outputNum)# 找到最大值的索引
    print('网络判断出来的数字是：%s，' % argmax)
    if argmax == int(input_list[0]):
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = np.asarray(scorecard)
print('准确率为：%s' % (scorecard_array.sum() / scorecard_array.size))
