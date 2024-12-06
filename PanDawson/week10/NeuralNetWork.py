import numpy
import scipy.special
import matplotlib.pyplot as plt


class  NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        '''
        初始化权重矩阵，我们有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵
        一个是who,表示中间层和输出层间链路权重形成的矩阵
        '''
        #生成一个形状为 (self.hnodes, self.inodes) 的数组  因为后面是wih * inputs
        self.wih = numpy.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = numpy.random.rand(self.onodes, self.hnodes) - 0.5

        '''
        scipy.special.expit对应的是sigmod函数.
        lambda是Python关键字，类似C语言中的宏定义，当我们调用self.activation_function(x)时，编译器会把其转换为spicy.special_expit(x)。
        '''
        self.activation_function = lambda x:scipy.special.expit(x)
        pass
    def  train(self, inputs_list, targets_list):
        #根据输入的训练数据更新节点链路权重
        '''
        把inputs_list, targets_list转换成numpy支持的二维矩阵
        .T表示做矩阵的转置
        '''
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        #计算信号经过输入层后产生的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        #中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        #输出层接收来自中间层的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activation_function(final_inputs)

        #计算loss
        output_errors = targets - final_outputs
        #公式计算反向梯度
        hidden_errors = numpy.dot(self.who.T, output_errors*final_outputs*(1-final_outputs))
        #根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * numpy.dot((output_errors * final_outputs *(1 - final_outputs)),
                                       numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                       numpy.transpose(inputs))
        pass
        
    def  query(self, inputs):
        #根据输入数据计算并输出答案
        #计算中间层从输入层接收到的信号量
        hidden_inputs = numpy.dot(self.wih, inputs)
        #计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        #计算最外层接收到的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs


data_file = open("dataset/mnist_test.csv")
data_list = data_file.readlines()
data_file.close()
#把数据依靠','区分，并分别读入——10张图片，每一行 28*28+1 = 785列。第一列是标号，后面28*28是像素
all_values = data_list[0].split(',')
#第一个值对应的是图片的表示的数字，所以我们读取图片数据时要去掉第一个数值
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))

onodes = 10
targets = numpy.zeros(onodes) + 0.01
targets[int(all_values[0])] = 0.99
print(targets)  #targets第8个元素的值是0.99，这表示图片对应的数字是7(数组是从编号0开始的).

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("dataset/mnist_train.csv")
trainning_data_list = training_data_file.readlines()
training_data_file.close()
#把数据依靠','区分，并分别读入
epochs = 10
for e in range(epochs):
    for record in trainning_data_list:
        all_values = record.split(',')
        #把所有数值全部转换到0.01到1.0之间。
        inputs = (numpy.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        #设置图片与数值的对应关系
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

scores = []
test_data_file = open("dataset/mnist_train.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:",correct_number)
    #预处理数字图片
    inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    #让网络判断图片对应的数字,推理
    outputs = n.query(inputs)
    #找到数值最大的神经元对应的 编号
    label = numpy.argmax(outputs)  
    print("output reslut is : ", label)
    #print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

#计算图片判断的成功率
scores_array = numpy.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)

# 该图片对应的数字为: 3
# [0.01467928 0.04850485 0.01147602 0.91671321 0.02615247 0.0044282
#  0.01074627 0.01050775 0.00584692 0.09501239]
# output reslut is :  3
# 该图片对应的数字为: 1
# [0.00172111 0.91991661 0.01538225 0.02417372 0.05930773 0.01221715
#  0.0070279  0.05664613 0.02862424 0.02121053]
# output reslut is :  1
# [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# perfermance =  0.99

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution() 
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise
 
#定义两个placeholder存放输入数据
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])
 
#定义神经网络中间层
Weights_L1=tf.Variable(tf.random_normal([1,20]))
biases_L1=tf.Variable(tf.zeros([1,20]))    #加入偏置项
Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1
L1=tf.nn.tanh(Wx_plus_b_L1)   #加入激活函数

#定义神经网络输出层
Weights_L2=tf.Variable(tf.random_normal([20,1]))
biases_L2=tf.Variable(tf.zeros([1,1]))  #加入偏置项
Wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2
prediction=tf.nn.tanh(Wx_plus_b_L2)   #加入激活函数

#定义损失函数（均方差函数）
loss=tf.reduce_mean(tf.square(y-prediction))
#定义反向传播算法（使用梯度下降算法训练）
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
 
with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    #训练2000次
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
 
    #获得预测值
    prediction_value=sess.run(prediction,feed_dict={x:x_data})
 
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)   #散点是真实值
    plt.plot(x_data,prediction_value,'r-',lw=5)   #曲线是预测值
    plt.show()


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                #将模型参数的梯度清零。因为在每次反向传播计算梯度之前，需要清除上一次迭代所积累的梯度，否则梯度会不断累加，导致错误的参数更新
                self.optimizer.zero_grad()

                # forward + backward + optimize
                #前向传播
                outputs = self.net(inputs)
                #计算损失值 loss
                loss = self.cost(outputs, labels)
                #反向传播
                loss.backward()
                #利用计算得到的梯度，按照优化器更新参数
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def mnist_load_data():
    #通过 transforms.Compose 将多个单独的变换操作组合在一起
    transform = transforms.Compose(
        #转换成张量格式 + 归一化（均值0 标准差1）
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
    return trainloader, testloader


class MnistNet(torch.nn.Module):
    def __init__(self):
        #super(Linear, self).__init__() 调用父类（torch.nn.Module）的构造函数，确保自定义模块能具备 torch.nn.Module 的基本功能
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        #第一维是批量大小（由 -1 自动推断），第二维是特征维度（28×28）
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

if __name__ == '__main__':
    # train for mnist
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)

