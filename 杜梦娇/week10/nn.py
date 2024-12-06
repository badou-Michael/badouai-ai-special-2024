import numpy as np
import scipy.special

#定义模型
class myModel:
    def __init__(self, inputsize, hiddensize, outputsize, lr):
        self.inlayer = inputsize
        self.hilayer = hiddensize
        self.oulayer = outputsize
        self.lr = lr
        #初始化参数（权重+偏置）
        self.w1 = np.random.rand(self.hilayer, self.inlayer) - 0.5
        self.w2 = np.random.rand(self.oulayer, self.hilayer) - 0.5

        #定义激活函数（sigmoid）
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T
        targets = np.array(targets, ndmin=2).T
        #正向传播
        hiddenlayer = self.activation_function(np.dot(self.w1, inputs))
        outputlayer =  self.activation_function(np.dot(self.w2, hiddenlayer))

        #误差计算+反向传播
        outputerror = targets - outputlayer
        hiddenerror = np.dot(self.w2.T, outputerror * outputlayer * (1-outputlayer))
        #更新权重
        self.w2 += self.lr * np.dot(outputerror * outputlayer * (1 - outputlayer), np.transpose(hiddenlayer))
        self.w1 += self.lr * np.dot(hiddenerror * hiddenlayer * (1 - hiddenlayer), np.transpose(inputs))

        pass

    def test(self, inputs):

        hiddenlayer = self.activation_function(np.dot(self.w1, inputs))
        outputlayer = self.activation_function(np.dot(self.w2, hiddenlayer))
        print(outputlayer)
        return outputlayer


#初始化模型
input_size = 28 * 28
hidden_size = 200
output_size = 10
learn_rate = 0.2
net = myModel(input_size, hidden_size, output_size, learn_rate)
#数据读取和预处理
train_data_file = open("dataset/mnist_train.csv")
train_data_list = train_data_file.readlines()
train_data_file.close()

#训练模型
epochs = 20
for e in range(epochs):
    for i in train_data_list:
        all_values = i.split(',')
        inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        #设置图片与数值的对应关系
        targets = np.zeros(10) + 0.01
        targets[int(all_values[0])] = 0.99
        net.train(inputs, targets)

#测试数据
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

scores = []
for i in test_data_list:
    all_values = i.split(',')
    print("该图片对应的数字为:", int(all_values[0]))
    inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
    outputs = np.argmax(net.test(inputs))
    print("output reslut is : ", outputs)
    #print("网络认为图片的数字是：", label)
    if outputs == int(all_values[0]):
        scores.append(1)
    else:
        scores.append(0)
print(scores)
# 计算准确率
accuracy = sum(scores) / len(scores)
print("Accuracy:", accuracy)
