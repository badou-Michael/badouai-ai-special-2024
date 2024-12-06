import numpy as np
from scipy.special import expit  # Sigmoid激活函数

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """
        初始化神经网络，定义层结构和学习率，并随机初始化权重。
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # 初始化权重矩阵，采用正态分布，均值为0，标准差为节点数的倒数平方根
        self.weights_input_hidden = np.random.normal(
            0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes)
        )
        self.weights_hidden_output = np.random.normal(
            0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes)
        )

        # 激活函数：Sigmoid
        self.activation_function = lambda x: expit(x)

    def train(self, input_list, target_list):
        """
        通过输入和目标值调整权重，训练神经网络。
        """
        # 将输入和目标值转换为二维数组并转置
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        # 前向传播：计算隐藏层和输出层的信号
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weights_hidden_output.T, output_errors)

        # 根据误差反向调整权重
        self.weights_hidden_output += self.learning_rate * np.dot(
            (output_errors * final_outputs * (1 - final_outputs)),
            hidden_outputs.T
        )
        self.weights_input_hidden += self.learning_rate * np.dot(
            (hidden_errors * hidden_outputs * (1 - hidden_outputs)),
            inputs.T
        )

    def query(self, input_list):
        """
        根据输入数据推断输出结果（前向传播）。
        """
        inputs = np.array(input_list, ndmin=2).T
        hidden_inputs = np.dot(self.weights_input_hidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.weights_hidden_output, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


def preprocess_data(data_list):
    """
    预处理数据，将像素值归一化到 [0.01, 1.0] 范围。
    """
    processed_data = []
    for record in data_list:
        all_values = record.split(',')
        label = int(all_values[0])  # 提取标签
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # 归一化像素值
        target = np.zeros(output_nodes) + 0.01
        target[label] = 0.99  # 设置目标值
        processed_data.append((inputs, target, label))
    return processed_data


def train_network(network, training_data, epochs):
    """
    训练神经网络。
    """
    for epoch in range(epochs):
        for inputs, targets, _ in training_data:
            network.train(inputs, targets)


def test_network(network, test_data):
    """
    测试神经网络，并计算准确率。
    """
    scores = []
    for inputs, _, correct_label in test_data:
        outputs = network.query(inputs)
        predicted_label = np.argmax(outputs)
        scores.append(1 if predicted_label == correct_label else 0)

    scores_array = np.asarray(scores)
    accuracy = scores_array.sum() / scores_array.size
    return accuracy


if __name__ == "__main__":
    # 初始化网络结构
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 加载并预处理训练数据
    with open("dataset/mnist_train.csv", 'r') as f:
        training_data_list = f.readlines()
    training_data = preprocess_data(training_data_list)

    # 加载并预处理测试数据
    with open("dataset/mnist_test.csv", 'r') as f:
        test_data_list = f.readlines()
    test_data = preprocess_data(test_data_list)

    # 训练网络
    epochs = 5
    train_network(nn, training_data, epochs)

    # 测试网络
    accuracy = test_network(nn, test_data)
    print(f"网络准确率: {accuracy * 100:.2f}%")