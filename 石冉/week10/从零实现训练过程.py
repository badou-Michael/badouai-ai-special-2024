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


        #初始化权重矩阵:wih表示输入层与中间层的权重矩阵，who表示中间层与输出层的权重矩阵
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))


        # 每个节点执行sigmoid激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        # 根据输入的训练数据更新节点链路权重
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # 计算第一个权重矩阵与输入矩阵相乘结果，作为中间层的输入
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 过激活函数
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算第二个权重矩阵与中间层矩阵相乘结果，作为输出层的输入
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 过激活函数后，得到输出结果
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 根据误差计算权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    def query(self, inputs):
        # 同train函数的前向训练
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs


# 初始化网络
# 由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读入训练数据
training_data_file = open("week10_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# 加入epochs,设定训练循环次数
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

test_data_file = open("week10_dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

# 计算图片判断的成功率
scores_array = numpy.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)
