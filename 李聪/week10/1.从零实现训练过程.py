import numpy as np
import scipy.special

class NeuralNetWork:
    """
    神经网络类，用于构建一个简单的神经网络。
    """
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """
        初始化神经网络参数。
        """
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate

        # 初始化权重矩阵：
        # wih：输入层到隐藏层的权重矩阵
        # who：隐藏层到输出层的权重矩阵
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 激活函数：sigmoid
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        """
        训练网络，更新权重。
        """
        # 将输入和目标数据转换为列向量
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 前向传播
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 反向传播
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        # 更新权重
        self.who += self.lr * np.dot(output_errors * final_outputs * (1.0 - final_outputs),
                                     hidden_outputs.T)
        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs),
                                     inputs.T)

    def query(self, inputs_list):
        """
        推理函数，根据输入获取输出。
        """
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

# 参数设置
input_nodes = 784  # 输入节点数
hidden_nodes = 200  # 隐藏层节点数
output_nodes = 10  # 输出节点数
learning_rate = 0.1  # 学习率

# 初始化网络
nn = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 模拟训练数据
train_inputs = np.random.rand(784)  # 归一化输入数据
train_targets = np.zeros(output_nodes) + 0.01
train_targets[3] = 0.99  # 设置目标为类别3

# 训练网络
nn.train(train_inputs, train_targets)

# 模拟测试数据
test_inputs = np.random.rand(784)  # 随机生成归一化测试输入
output = nn.query(test_inputs)

# 输出结果
print("预测输出：", output)
print("预测类别：", np.argmax(output))
