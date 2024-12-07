import numpy as np
import scipy.special

class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 初始化网络，设置输入层、中间层、输出层节点
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # 设置学习率
        self.lr = learningrate

       # 初始化权重矩阵
       # 两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
       # 一个是who,表示中间层和输出层间链路权重形成的矩阵
        self.wih = np.random.rand(self.hnodes,self.inodes)-0.5
        self.who = np.random.rand(self.onodes,self.hnodes)-0.5

        '''
        每个节点执行激活函数，得到的结果将作为信号输出到下一层，我们用sigmoid作为激活函数
        '''
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # 训练模型，输入列表和标签
    # 根据输入的训练数据更新节点链路权重
    def train(self, inputs_list, targets_list):
        # 把inputs_list, targets_list转换成numpy支持的二维矩阵，.T表示做矩阵的转置
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T

        # 计算输入层经过权重矩阵wih后，到隐藏层的输入信号量
        hidden_inputs = np.dot(self.wih,inputs)
        # 计算隐层层输入信号量，经过激活函数后的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算隐藏层经过权重矩阵who后，到输出层的输入信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层的输入信号量，经过激活函数后的最终输出信号量
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets -final_outputs
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        np.transpose(inputs))
        pass
    # 推理预测
    def  query(self,inputs):
        #根据输入数据计算并输出答案
        #计算中间层从输入层接收到的信号量
        hidden_inputs = np.dot(self.wih, inputs)
        #计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        #计算最外层接收到的信号量
        final_inputs = np.dot(self.who, hidden_outputs)
        #计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs

#初始化网络
'''
csv文件中，共100行，每行785个元数，第1位是标签值
剩余784位是训练集，28*28
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
'''
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n= NeuralNetWork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#读入训练数据
training_data_file = open("dataset/mnist_train.csv",'r')
# 按行读取形成list,每个元素是字符串 用，隔开
training_data_list = training_data_file.readlines()
print(training_data_list)
training_data_file.close()

#加入epocs,设定网络的训练循环次数
epochs = 5
for e in range(epochs):
    #把数据依靠','区分，并分别读入
    for record in training_data_list:
        all_values = record.split(',')
        # asfarray转换为浮点数，归一化处理
        inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        #设置图片与数值的对应关系
        targets = np.zeros(output_nodes) + 0.01
        # 索引值第几位，值为0.99，其余为0
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

# 测试集进行测试
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
# 与上述同理
for record in test_data_list:
    all_values = record.split(',')

    #预处理数字图片，转化为浮点数后，进行归一化
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    #让网络判断图片对应的数字
    outputs = n.query(inputs)
    #找到数值最大的神经元对应的编号
    label = np.argmax(outputs)
    correct_number = int(all_values[0])
    print("该图片对应的准确数字为:", correct_number)
    print("网络预测图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

#计算图片判断的成功率
scores_array = np.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)
