import numpy as np
import scipy.special
class NeuralNetWork:
    # 初始化神经网络
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        '''
        这里的随机数矩阵形状顺序不要搞错，后面np.dot乘法需要矩阵相乘，要定义清楚
        否则后面的矩阵乘法会报错
        '''
        self.wih = np.random.rand(self.hnodes, self.inodes)-0.5
        # 200 * 784矩阵
        # print(self.wih.shape)
        self.who = np.random.rand(self.onodes, self.hnodes)-0.5
        # 10 * 200矩阵
        # print(self.who.shape)
        self.activation_function = lambda x :scipy.special.expit(x)
        pass
    def train(self,inputs_list, targets_list):
        # 转为二维矩阵不改变数据情况下进行矩阵运算
        inputs = np.array(inputs_list, ndmin = 2).T
        # inputs是784*1矩阵
        # print(inputs.shape)
        targets = np.array(targets_list, ndmin = 2).T
        # 计算信号经过输入层后产生的信号量
        hidden_inputs = np.dot(self.wih, inputs)
        # 激活函数
        hidden_outputs = self.activation_function(hidden_inputs)
        # 隐藏层到输出层的信号量
        # hidden_outputs是200*1矩阵
        # print(hidden_inputs.shape)
        final_inputs = np.dot(self.who,hidden_outputs)
        # 激活函数
        final_outputs = self.activation_function(final_inputs)
        # 计算误差
        output_errors = targets - final_outputs
        # 参照误差的计算公式
        hidden_errors = np.dot(self.who.T, output_errors*final_outputs*(1-final_outputs))
        # 反向传播更新权重
        self.who += self.lr * np.dot((output_errors * final_outputs * (1-final_outputs)),
                                   np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs *(1-hidden_outputs)),
                                   np.transpose(inputs))
        pass
    def query(self,inputs):
        # 计算输入层==》隐藏层的信号量，dot点积
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算隐藏层==》输出层的信号量，dot点积
        final_inputs = np.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs
# 神经网络参数设置
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.3
n = NeuralNetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)

# 训练数据读取
training_data_file = open(r'task10_detail/dataset/mnist_train.csv')
training_data_list = training_data_file.readlines()
training_data_file.close()
# 设定数据集训练循环次数
epoches = 5
for e in range(epoches):
    for record in training_data_list:
        all_values = record.split(',')
        # 归一化处理
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 设置图片与数值的对应关系
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        # 调用train函数
        n.train(inputs, targets)
test_data_file = open('task10_detail/dataset/mnist_test.csv')
test_data_list = test_data_file.readlines()

test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    # 测试数据中第一个数值为手写数字，定义为识别变量
    correct_number = int(all_values[0])
    print('该图片对应的数字为：',correct_number)
    inputs = (np.asfarray(all_values[1:])) /255.0 * 0.99 +0.01
    # 调用函数判断图片对应的数字
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    print('网络认为图片的数字为，',label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
# 计算图片的成功率
scores_array = np.asfarray(scores)
print('perfermance = ',scores_array.sum() / scores_array.size)
