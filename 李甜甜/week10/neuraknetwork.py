# 先搭建框架
# 初始化
import numpy as np
import scipy.special

class NeuralNetWork:
    def __init__(self, inputnodes, outputnodes,hiddennodes,learningrate):
        """初始化，InputNodes, OutputNodes,HiddenNodes,learningRate"""
        self.InNodes = inputnodes
        self.OutNodes = outputnodes
        self.HNodes = hiddennodes
        self.lr = learningrate
        self.wih = np.random.random((inputnodes, hiddennodes))-0.5
        self.who = np.random.random((hiddennodes, outputnodes))-0.5
        #激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, train_data, label):
        """训练, 大部分和推理一样，但训练数据和标签要变成一个二维矩阵，以便和后面二维权重进行点乘"""
        train_data = np.array(train_data, ndmin=2).T
        label = np.array(label, ndmin=2).T
        print(train_data, label)
        hiddenin = np.dot(self.wih, train_data)
        # 隐藏层输入经过激活函数得到隐藏层输出结果
        hiddenout = self.activation_function(hiddenin)
        # 隐藏层输出经过点乘得到输出节点的输入值
        output1 = np.dot(self.who, hiddenout)
        # 经过激活函数得到输出值
        output2 = self.activation_function(output1)
        # 计算损失函数
        output_error = label - output2
        print('output_error', output_error)
        print('output2', output2)

        # 隐藏层输出值对损失函数的影响, 注意反向矩阵要转置
        hidden_error = np.dot(self.who.T, output_error*output2*(1-output2))
        print(hidden_error)
        # 更新权重
        print('1')
        print(output_error*output2*(1-output2))
        print(hiddenout*(1-hiddenout)*hidden_error)
        self.who += self.lr*np.dot(output_error*output2*(1-output2), np.transpose(hiddenout))
        self.wih += self.lr*np.dot(hiddenout*(1-hiddenout)*hidden_error, np.transpose(train_data))

    def query(self, testdata):
        """测试"""
        # 输入层经过点乘，得到隐藏层的输入
        hiddenin = np.dot(self.wih, testdata)
        # 隐藏层输入经过激活函数得到隐藏层输出结果
        hiddenout = self.activation_function(hiddenin)
        # 隐藏层输出经过点乘得到输出节点的输入值
        output1 = np.dot(self.who, hiddenout)
        # 经过激活函数得到输出值
        output2 = self.activation_function(output1)
        print(output2)
        return output2

# 中途测试 初始化和正向传播 测试无误

inodes = 3
onodes = 3
hnode = 3
lr = 0.3
n = NeuralNetWork(inodes, onodes, hnode, lr)
print(n.InNodes)
print(n.who)
testd = [0.1, 0.5, 0.6]
pridect_class = n.query(testd)
target = [1, 0, 0]
# 测试数据转换
print(n.train(testd, target))
