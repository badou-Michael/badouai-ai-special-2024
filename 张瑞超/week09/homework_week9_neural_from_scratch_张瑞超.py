import numpy
import scipy.special

# 定义神经网络类
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 初始化网络节点数和学习率
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate

        # 初始化权重矩阵，使用均匀分布，中心点在0
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 激活函数为sigmoid函数
        self.activation_function = lambda x: scipy.special.expit(x)

    # 训练函数，更新权重
    def train(self, inputs_list, targets_list):
        # 将输入和目标列表转换为二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 计算隐藏层的输入和输出
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算输出层的输入和输出
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算输出层误差
        output_errors = targets - final_outputs
        # 反向传播计算隐藏层误差
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # 更新权重
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)), hidden_outputs.T)
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), inputs.T)

    # 查询函数，根据输入进行推理
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 计算隐藏层的输入和输出
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算输出层的输入和输出
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# 加载和处理MNIST数据集
def load_data(filepath):
    with open(filepath, 'r') as data_file:
        data_list = data_file.readlines()
    return data_list

# 数据预处理，将图像数据归一化
def preprocess_data(record, output_nodes):
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    return inputs, targets

# 计算网络性能，包含识别正确性和识别的准确率
def evaluate_performance(nn, test_data_list):
    scores = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_number = int(all_values[0])
        print(f"该图片对应的数字为: {correct_number}")

        # 数据预处理
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

        # 推理结果
        outputs = nn.query(inputs)
        label = numpy.argmax(outputs)

        print(f"网络认为图片的数字是：{label}")

        # 判断是否正确
        if label == correct_number:
            scores.append(1)
        else:
            scores.append(0)

    print(f"识别结果：{scores}")
    return numpy.asarray(scores)

# 初始化神经网络参数
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.3

# 创建神经网络实例
nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 训练数据集路径
train_data_path = "dataset/mnist_train.csv"
train_data_list = load_data(train_data_path)

# 训练神经网络
epochs = 10
for e in range(epochs):
    for record in train_data_list:
        inputs, targets = preprocess_data(record, output_nodes)
        nn.train(inputs, targets)

# 测试数据集路径
test_data_path = "dataset/mnist_test.csv"
test_data_list = load_data(test_data_path)

# 评估模型性能
scores_array = evaluate_performance(nn, test_data_list)
print(f"Performance: {scores_array.sum() / scores_array.size:.4f}")
