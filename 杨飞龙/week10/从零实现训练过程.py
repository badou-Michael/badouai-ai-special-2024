import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 初始化输入层、隐藏层、输出层节点数
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # 初始化权重矩阵，使用正态分布随机数进行初始化
        self.weights_input_hidden = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.weights_hidden_output = np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        # 激活函数：Sigmoid函数
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, input_list, target_list):
        # 将输入数据转换为二维数组
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # 正向传播
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)  # 输入层到隐藏层的输入
        hidden_outputs = self.activation_function(hidden_inputs)  # 隐藏层的输出

        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)  # 隐藏层到输出层的输入
        final_outputs = self.activation_function(final_inputs)  # 最终输出层的输出

        # 计算误差
        output_errors = targets - final_outputs  # 输出层误差
        hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)  # 隐藏层误差

        # 反向传播：更新权重
        self.weights_hidden_output += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs.T)
        self.weights_input_hidden += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), inputs.T)

    def query(self, input_list):
        # 将输入数据转换为二维数组
        inputs = np.array(input_list, ndmin=2).T

        # 正向传播
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)  # 输入层到隐藏层的输入
        hidden_outputs = self.activation_function(hidden_inputs)  # 隐藏层的输出

        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)  # 隐藏层到输出层的输入
        final_outputs = self.activation_function(final_inputs)  # 最终输出层的输出

        return final_outputs

# 示例用法
if __name__ == "__main__":
    # 定义输入层、隐藏层、输出层节点数，以及学习率
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3

    # 创建神经网络实例
    nn = SimpleNeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 使用示例数据训练网络
    for _ in range1000):
        nn.train([1.0, 0.5, -1.5], [0.1, 0.9, 0.1])

    # 查询网络输出
    output = nn.query([1.0, 0.5, -1.5])
    print("网络输出:", output)
