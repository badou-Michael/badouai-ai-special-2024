import numpy
import scipy.special

class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):     # 参数初始化
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))
        self.activation_function = lambda x: scipy.special.expit(x)     # 匿名函数语法
        pass

    def train(self, inputs_list, targets_list):     # 训练
        # 数据预处理
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # 计算预激活函数
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 激活函数
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算信号输出
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 基于梯度损失调整神经网络权值
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    def query(self, inputs):    # 推理
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs

# 设置各项参数的初始值
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)   # 实例化神经网络模型
# 读入训练数据
training_data_file = open("dataset/mnist_train.csv", 'r')   # 'r'表示以只读模式打开文件
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')  # 用逗号将分类标签值和像素点值分隔开
        # 数据预处理
        inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01   # all_values[1:]代表像素值
        # 设置图片标签与输出概率值的对应关系—即进行one hot编码
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        # 进行训练
        n.train(inputs, targets)

# 读取测试数据
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])     # 提取测试记录中的正确数字的索引标签
    print("该图片对应的数字为:", correct_number)
    # 数据预处理
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    # 进行推理
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
