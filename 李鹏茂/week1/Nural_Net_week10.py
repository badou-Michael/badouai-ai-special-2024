import numpy
import scipy.special
import math
class NeuralNet:
    def __init__(self,input_nodes,hidden_nodes,output_nodes,learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.wih = (numpy.random.normal(0.0, pow(self.hidden_nodes,-0.5), (self.hidden_nodes,self.input_nodes) )  )
        self.who = (numpy.random.normal(0.0, pow(self.output_nodes,-0.5), (self.output_nodes,self.hidden_nodes) )  )
        # self.activation_function = lambda x: scipy.special.expit(x)
        self.activation_function = lambda x: 1 / (1 +  numpy.exp(-x))
    def forward(self,inputs):

        hidden_inputs = numpy.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    def backward(self, inputs,targets):
        inputs = numpy.array(inputs, ndmin=2).T

        targets = numpy.array(targets, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        # print(self.wih.shape,inputs.shape,hidden_inputs.shape, targets.shape)

        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)
        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # print(hidden_errors.shape,hidden_outputs.shape,output_errors.shape,(output_errors * final_outputs * (1 - final_outputs)).shape)
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        # print(numpy.transpose(hidden_outputs).shape,final_outputs.shape,(output_errors * final_outputs * (1)))
        self.who += self.learning_rate * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                                   numpy.transpose(hidden_outputs))
        # print(self.who.shape)
        self.wih += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNet(input_nodes, hidden_nodes, output_nodes, learning_rate)

#读入训练数据
#open函数里的路径根据数据存储的路径来设定
training_data_file = open("dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#加入epocs,设定网络的训练循环次数
epochs = 5
for e in range(epochs):
    #把数据依靠','区分，并分别读入
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        #设置图片与数值的对应关系
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99

        n.backward(inputs, targets)

test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:",correct_number)
    #预处理数字图片
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01

    #让网络判断图片对应的数字
    outputs = n.forward(inputs)
    #找到数值最大的神经元对应的编号
    label = numpy.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)
