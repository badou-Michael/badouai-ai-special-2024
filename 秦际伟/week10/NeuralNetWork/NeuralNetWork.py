import numpy
import scipy.special

class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化网络，设置输入层，中间层，和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 设置学习率
        self.lr = learningrate
        # 输入层和隐藏层的权重矩阵
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # 隐藏层和输出层的权重矩阵
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # 激活函数，使用sigmoid函数
        self.activation_function = lambda x:scipy.special.expit(x)
        # 激活函数，使用relu函数
        # self.activation_function = lambda x: numpy.maximum(0, x)


    # 训练网络  根据输入的训练样本，更新权重矩阵
    def train(self, inputs_list, targets_list):
        # 转换输入列表为二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 计算隐藏的输入值
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors*final_outputs*(1-final_outputs))

        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * numpy.dot((output_errors * final_outputs *(1 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), numpy.transpose(inputs))


    # 根据输入数据计算输出
    def query(self, inputs_list):
        # 转换输入列表为二维数组
        inputs = numpy.array(inputs_list, ndmin=2).T
        # 计算隐藏的输入值
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

# 设置参数
input_nodes = 784
hidden_nodes = 500
output_nodes = 10
learning_rate = 0.2
epochs = 5
net = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# 读取训练数据
training_data_file = open("dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        net.train(inputs, targets)


# 读取测试数据
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = net.query(inputs)
    label = numpy.argmax(outputs)
    if label == correct_label:
        scores.append(1)
    else:
        scores.append(0)
    print("正确数字：", correct_label, "预测数字：", label)
print("正确次数：", scores.count(1))

# 计算图片判断的成功率
scores_array = numpy.asarray(scores)
print("正确率：", scores_array.sum() / scores_array.size)
