import numpy as np
from scipy.special import expit


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        """
        初始化神经网络。
        :param input_size: 输入层节点数量
        :param hidden_size: 隐藏层节点数量
        :param output_size: 输出层节点数量
        :param learning_rate: 学习率
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 初始化权重矩阵，采用均值为0，标准差与节点数成反比的正态分布
        self.weights_input_hidden = np.random.normal(0.0, hidden_size ** -0.5, (hidden_size, input_size))
        self.weights_hidden_output = np.random.normal(0.0, output_size ** -0.5, (output_size, hidden_size))

        # 使用sigmoid作为激活函数
        self.activation_function = expit

    def train(self, input_data, target_data):
        """
        根据输入数据和目标值更新权重。
        :param input_data: 输入数据
        :param target_data: 目标数据
        """
        inputs = np.array(input_data, ndmin=2).T
        targets = np.array(target_data, ndmin=2).T

        # 正向传播
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_hidden_output.T, output_errors * final_outputs * (1 - final_outputs))

        # 更新权重
        self.weights_hidden_output += self.learning_rate * np.dot(
            (output_errors * final_outputs * (1 - final_outputs)), hidden_outputs.T
        )
        self.weights_input_hidden += self.learning_rate * np.dot(
            (hidden_errors * hidden_outputs * (1 - hidden_outputs)), inputs.T
        )

    def query(self, input_data):
        """
        根据输入数据计算输出。
        :param input_data: 输入数据
        :return: 网络输出
        """
        inputs = np.array(input_data, ndmin=2).T

        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# 设置网络结构参数
input_size = 784
hidden_size = 200
output_size = 10
learning_rate = 0.1

# 创建神经网络实例
network = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

# 加载训练数据
with open("dataset/mnist_train.csv", 'r') as file:
    training_data = file.readlines()

epochs = 5
for _ in range(epochs):
    for record in training_data:
        #把数据依靠','区分，并分别读入
        values = record.split(',')
        inputs = (np.asfarray(values[1:]) / 255.0) * 0.99 + 0.01
        
        '''
        作用是对输入值进行归一化处理，并将归一化的范围限定在 [0.01, 1.0] 之间
        
        为什么是 0.99 和 0.01?
        避免极端值的影响：

        如果直接将像素值归一化到 [0, 1]，激活函数 (如 Sigmoid) 的输入会有可能接近极限值 0 或 1, 
        导致梯度过小（梯度消失问题），从而影响学习效果。

        通过限制范围为 [0.01, 1.0]，确保输入值不会太接近激活函数的极限，保留一定的梯度，增强学习能力。
        缩放到 [0.01, 1.0] 的实现：

        像素值通常是 0-255 的整数。
        (np.asfarray(values[1:]) / 255.0) 将像素值归一化到 [0, 1]。
        再乘以 0.99, 将范围扩展到 [0, 0.99]。
        加上 0.01, 最终映射到 [0.01, 1.0]。
        '''
        
        #设置图片与数值的对应关系
        targets = np.zeros(output_size) + 0.01
        targets[int(values[0])] = 0.99
        network.train(inputs, targets)

# 加载测试数据
with open("dataset/mnist_test.csv", 'r') as file:
    test_data = file.readlines()

# 测试网络性能
correct_predictions = []
for record in test_data:
    values = record.split(',')
    true_label = int(values[0])
    print("该图片对应的数字为:", true_label)
    
    #预处理数字图片
    inputs = (np.asfarray(values[1:]) / 255.0) * 0.99 + 0.01
    
    #让网络判断图片对应的数字
    outputs = network.query(inputs)
    
    #找到数值最大的神经元对应的编号
    predicted_label = np.argmax(outputs)
    print("网络认为图片的数字是：", predicted_label)

    correct_predictions.append(1 if predicted_label == true_label else 0)

# 输出模型性能
accuracy = np.mean(correct_predictions)
print(f"Network accuracy: {accuracy:.2%}")

