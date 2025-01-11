#-*- coding:utf-8 -*-
# author: 王博然
import numpy as np
import scipy.special

class NeuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # 初始化网络: 设置输入层、中间层 和 输出层节点数
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes
        self.lr = learningRate
        # 初始化两个权重矩阵, 由于权重不一定都是正的，它完全可以是负数
        self.wih = np.random.random((self.hnodes, self.inodes)) - 0.5
        self.who = np.random.random((self.onodes, self.hnodes)) - 0.5
        # 激活函数, 设置为 sigmoid
        self.activation_function = lambda x:scipy.special.expit(x)

    def train(self, inputs_list, targets_list): # 训练: 根据训练数据更新节点链路权重
        # 把inputs_list 和 targets_list 转成 numpy支持的二维矩阵
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 1.1 计算中间层从输入层接收到的信号量
        hidden_inputs = np.dot(self.wih, inputs)  # [hide,input] * [input,1] = [hide, 1]
        # 1.2 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)

        # 2.1 计算最外层接收到的信号量
        final_inputs = np.dot(self.who, hidden_outputs) # [out, hide] * [hide, 1] = [out, 1]
        # 2.2 计算最外层经过激活函数后的信号量
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1-final_outputs))
        # 更新权重 (负负得正)
        self.who += self.lr * np.dot(output_errors * final_outputs * (1 - final_outputs), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), np.transpose(inputs))

    def query(self, inputs): # input 格式待确认
        # 推理: 根据输入数据计算并输出答案
        # 1.1 计算中间层从输入层接收到的信号量
        hidden_inputs = np.dot(self.wih, inputs)  # [hide,input] 
        # 1.2 计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 2.1 计算最外层接收到的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        # 2.2 计算最外层经过激活函数后的信号量
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

if __name__ == '__main__':
    input_nodes = 784   # 图片大小 28 * 28
    hidden_nodes = 100  # 随便选的经验值
    output_nodes = 10
    learning_rate = 0.3
    epochs = 5
    nw = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读入训练数据
    training_data_file = open("dataset/mnist_train.csv")
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')   # csv文件是以 ',' 作为分隔符的
            inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01  # 防止都是0, 导致链路权重更新出问题
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            nw.train(inputs, targets)

    # 读入测试数据
    test_data_file = open("dataset/mnist_test.csv")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scores = []
    for record in test_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        outputs = nw.query(inputs)
        # 找到最大值的编号 one-hot
        label = np.argmax(outputs)
        if label == int(all_values[0]):
            scores.append(1)
        else:
            scores.append(0)

    # 计算成功率
    print(scores)
    print("performance = %f" % (sum(scores)/len(scores)))