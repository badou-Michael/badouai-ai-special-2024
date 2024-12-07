import numpy as np 
import scipy.special
from matplotlib import pyplot as plt
import cv2

class Network():
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        #初始化网络结构
        self.innodes = inputnodes
        self.hinodes = hiddennodes
        self.outnodes = outputnodes
        self.lr = learningrate

        #初始化权重
        self.wih = np.random.rand(self.innodes, self.hinodes) - 0.5
        self.who = np.random.rand(self.hinodes, self.outnodes) - 0.5

        #初始化激活函数
        self.activationfunction = lambda x: scipy.special.expit(x)


    def loss_function(self, outputs, target):
        # MSE 损失函数
        return sum((outputs-target)*(outputs-target)/2)
    
    def train(self, train_input, target):

        train_input = np.array(train_input, ndmin=2).T
        target = np.array(target, ndmin=2).T

        # print(train_input.shape)

        #正向传播
        hidden_outputs = self.activationfunction(np.dot(self.wih.T, train_input))
        final_outputs = self.activationfunction(np.dot(self.who.T, hidden_outputs))   
        loss = self.loss_function(final_outputs, target)
        # print("loss:", loss)

        #反向传播
        total_errors = target - final_outputs
        hidden_errors = np.dot(self.who, total_errors * final_outputs * (1 - final_outputs))

        # test1 = total_errors * final_outputs *(1 - final_outputs)
        # print(test1.shape)
        # print('隐藏层的shape:', hidden_outputs.shape)
        # print('隐藏层转置的shape:', (np.transpose(hidden_outputs)).shape)
        self.who += self.lr * np.dot((total_errors * final_outputs *(1 - final_outputs)), hidden_outputs.T).T
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), train_input.T).T
        
        return loss

    

    def query(self, test_input):

        # test_input = np.array(test_input, ndmin=2).T
        hidden_outputs = self.activationfunction(np.dot(self.wih.T, test_input))
        final_outputs = self.activationfunction(np.dot(self.who.T, hidden_outputs))
        print("final out:", final_outputs)
        return final_outputs
        

data = open('/home/zzy/桌面/pytest/dataset/mnist_train.csv')
datalist = data.readlines()
data.close()
# # print(type(datalist), len(datalist))
# img = np.asfarray(datalist[0].split(','))
# # print(type(img), img.shape)
# img = img[1:].reshape(28,18)
# print(img.shape)
# plt.imshow(img)
# plt.show()

inputnodes = 784
hiddenodes = 200
outputnodes = 10
learning_rate = 0.1
net = Network(inputnodes, hiddenodes, outputnodes, learning_rate)
epoch = 100
train_loss = np.array([])

for e in range(epoch):
    for lines in datalist:
        line = np.asfarray(lines.split(','))
        inputs = line[1:].reshape(-1)/255.0 * 0.99 + 0.01
        # print('input shape:', inputs.shape)
        target = np.zeros(outputnodes) + 0.01
        target[int(line[0])] = 0.99
        # print('target shape:', target.shape)
        loss = net.train(inputs, target)
        train_loss = np.append(train_loss, loss)

print(train_loss.shape)

testdata = open('/home/zzy/桌面/pytest/dataset/mnist_test.csv')
test_list = testdata.readlines()
testdata.close()


correctnums = 0
# final_loss = np.array([])
for line in test_list:
    line = np.asfarray(line.split(','))
    test_input = line[1:].reshape(-1)/255.0 * 0.99 + 0.01
    label = int(line[0])
    print('正确的 图片是：', label)
    out = net.query(test_input)
    out_label = np.argmax(out)
    test_target = np.zeros(outputnodes) + 0.01
    test_target[out_label] = 0.99
    print('推理的结果为：', out_label)    
    if out_label == label:
        correctnums += 1
    # loss = net.loss_function(out, test_target)
    # print('loss:', loss)
    

print('正确率:', correctnums)

x_label = np.arange(len(train_loss))
# print(x_label)
plt.plot(x_label, train_loss)
plt.show()


test_img = cv2.imread('/home/zzy/桌面/pytest/dataset/my_own_2.png' ,cv2.IMREAD_GRAYSCALE)
cv2.imshow('2',test_img)
cv2.waitKey(0)
# print(test_img)
# print(test_img.shape)
test_img = test_img /255 * 0.99 + 0.01
test_in = test_img.reshape(-1)
predict = net.query(test_in)
res = np.argmax(predict)
print(res)




    