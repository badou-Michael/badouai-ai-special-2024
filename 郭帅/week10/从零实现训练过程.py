import numpy    # NumPy 是 Python 中用于进行高效数值计算和处理的一个库
import scipy.special    # SciPy 库中的 special 模块,包含了一些常用的特殊数学函数,通常使用的是 scipy.special.expit(x) 来计算 sigmoid 激活函数

'''
1、创建神经网络模型
'''
class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes    #设置输入层
        self.hnodes = hiddennodes   #设置隐藏层
        self.onodes = outputnodes   #设置输出层
        self.lr = learningrate      #设置学习率
        '''
        设置初始权重，numpy.random.normal 是 NumPy 库中用于生成正态分布（高斯分布）随机数的函数
        wih 是输入层到隐藏层的权重，权重矩阵（行数 = 隐藏层神经元数，列数 = 输入层神经元数）
        who 是隐藏层到输出层的权重，权重矩阵（行数 = 输出层神经元数，列数 = 隐藏层神经元数）
        每一层的标准差的确是由 下一层的神经元数 来决定的，pow(self.hnodes, -0.5) 表示 倒数平方根：1 / sqrt(self.hnodes) 。
        '''
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        self.activation_function = lambda x: scipy.special.expit(x)    #设置激活函数

        pass

    def train(self, inputs_list, targets_list):                     # inputs_list 和 targets_list 是输入数据和目标数据

        '''训练过程：前向传播（正向传播）'''
        inputs = numpy.array(inputs_list, ndmin=2).T                # ndmin=2 转换为 二维矩阵  .T 是矩阵的 转置操作，将行列交换
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)                 # 隐藏层的输入数据：输入层*权重矩阵
        hidden_outputs = self.activation_function(hidden_inputs)    # 隐藏层的输出数据：经过激活函数的数据
        final_inputs = numpy.dot(self.who, hidden_outputs)          # 输出层的输入数据：隐藏层*权重矩阵
        final_outputs = self.activation_function(final_inputs)      # 输出层的输出数据：经过激活函数的数据

        '''训练过程：反向传播，更新权重'''
        output_errors = targets - final_outputs                     # 计算误差 = 目标数据 - 输出数据
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))   # 计算隐藏层误差

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),       # 更新输出层权重
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),     # 更新隐藏层权重
                                        numpy.transpose(inputs))
        pass

    def  query(self,inputs):
        '''预测过程：根据输入数据计算并输出答案'''
        hidden_inputs = numpy.dot(self.wih, inputs)                  # 隐藏层的输入数据：输入层*权重矩阵
        hidden_outputs = self.activation_function(hidden_inputs)     # 隐藏层的输出数据：经过激活函数的数据
        final_inputs = numpy.dot(self.who, hidden_outputs)           # 输出层的输入数据：隐藏层*权重矩阵
        final_outputs = self.activation_function(final_inputs)       # 输出层的输出数据：经过激活函数的数据
        print(final_outputs)
        return final_outputs                                         # 最后返回 final_outputs，即神经网络的最终预测结果


'''
2、初始化网络（训练）
'''
# 由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)   #Python 的函数调用机制是基于位置参数的匹配，而不是基于参数名的匹配

#读入训练数据
training_data_file = open("dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()   #使用 readlines() 方法将文件内容读取到一个列表中，每一行数据会作为列表的一个元素
training_data_file.close()

#加入epochs,设定网络的训练循环次数
epochs = 5
for e in range(epochs):
    for record in training_data_list:           # 遍历所有训练数据,record 是列表中的一项（每一行数据）
        all_values = record.split(',')          # all_values 会是一个包含标签和像素值的列表，按逗号分割开
        inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01   # 对输入数据做归一化处理，all_values[1:] 取除标签之外的所有元素，即图片的像素值。
        targets = numpy.zeros(output_nodes) + 0.01  # 创建一个大小为 output_nodes 的数组，并将所有值初始化为 0.01
        targets[int(all_values[0])] = 0.99      # 将标签对应的位置设置为 0.99
        n.train(inputs, targets)

'''
3、测试过程
'''
test_data_file = open("dataset/mnist_test.csv")   # 打开测试数据集
test_data_list = test_data_file.readlines()       # 将测试集的内容读取到一个列表
test_data_file.close()
scores = []                                       # scores 用来存储每一张图片是否预测正确。正确预测时会添加 1，错误预测时添加 0
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])           # 从 all_values 中提取出标签并转换为整数，即图片的真实数字。
    print("该图片对应的数字为:",correct_number)
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    outputs = n.query(inputs)                     # 让网络判断图片对应的数字
    label = numpy.argmax(outputs)                 # 返回输出数组中最大值的索引，索引最大的神经元对应的数字即为网络预测的数字标签。
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

'''
4、计算图片判断的成功率
'''
accuracy = sum(scores) / len(scores) * 100
print(f"网络的准确率为：{accuracy}%")
