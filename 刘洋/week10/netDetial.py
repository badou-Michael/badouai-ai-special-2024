import numpy
import scipy.special


class MyNet:
    def __init__(self,inputnodes,hiddennodes,outputnodes,lr):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = lr

        # 初始化权重
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        # 设置激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        '''
        把inputs_list, targets_list转换成numpy支持的二维矩阵
        '''
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors*final_outputs*(1-final_outputs))
        # 权重更新
        self.who += self.lr * numpy.dot((output_errors * final_outputs *(1 - final_outputs)),
                                       numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                       numpy.transpose(inputs))

    def forward(self, inputs):
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

# 模型初始化
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
mynet = MyNet(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 导入训练集
training_data_file = open("dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
# 导入测试集
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

# 训练模型
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asarray(all_values[1:], dtype=numpy.float32))/255.0 * 0.99 + 0.01   # 归一化且【0.01-1】
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        mynet.train(inputs, targets)

# 测试模型
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:", correct_number)
    # 预处理
    inputs = (numpy.asarray(all_values[1:], dtype=numpy.float32)) / 255.0 * 0.99 + 0.01
    # 模型输出
    outputs = mynet.forward(inputs)
    # 输出转化
    label = numpy.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

scores_array = numpy.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)
