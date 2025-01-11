# -*- coding: utf-8 -*-
# time: 2024/11/14 16:55
# file: mnistnet.py
# author: flame
# -*- coding: utf-8 -*-
# time: 2024/11/14 16:55
# file: mnistnet.py
# author: flame
# -*- coding: utf-8 -*-
# time: 2024/11/14 16:55
# file: mnistnet.py
# author: flame
'''
    本代码实现了一个简单的全连接神经网络，用于MNIST手写数字识别任务。首先定义了网络结构MnistNet，
    然后定义了一个通用的模型类Model，包含网络、损失函数和优化器。接着实现了加载MNIST数据集的函数mnist_load_data，
    最后在主函数中实例化网络和模型，加载数据集，训练模型并评估其性能。
'''

''' 导入日志模块，用于记录训练过程中的信息。 '''
import logging

''' 导入PyTorch框架，用于构建和训练神经网络。 '''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

''' 定义一个简单的全连接神经网络，用于MNIST数字识别。 '''
class MnistNet(torch.nn.Module):
    ''' 初始化网络结构。 '''
    def __init__(self):
        super(MnistNet, self).__init__()
        ''' 定义第一个全连接层，输入维度为28*28（MNIST图像大小），输出维度为512。 '''
        self.fc1 = nn.Linear(28 * 28, 512)
        ''' 定义第二个全连接层，输入维度为512，输出维度为512。 '''
        self.fc2 = nn.Linear(512, 512)
        ''' 定义第三个全连接层，输入维度为512，输出维度为10（MNIST类别数）。 '''
        self.fc3 = nn.Linear(512, 10)

    ''' 定义网络的前向传播过程。 '''
    def forward(self, x):
        ''' 将输入图像展平为一维向量。 '''
        x = x.view(-1, 28 * 28)
        ''' 使用ReLU激活函数处理第一个全连接层的输出。 '''
        x = F.relu(self.fc1(x))
        ''' 使用ReLU激活函数处理第二个全连接层的输出。 '''
        x = F.relu(self.fc2(x))
        ''' 使用Softmax激活函数处理第三个全连接层的输出，得到类别概率。 '''
        x = F.softmax(self.fc3(x), dim=1)
        return x

''' 定义一个通用的模型类，包含网络、损失函数和优化器。 '''
class Model():
    ''' 初始化模型，包括网络结构、损失函数和优化器。 '''
    def __init__(self, net, cost, optimist):
        self.net = net
        ''' 创建损失函数。 '''
        self.cost = self.create_cost(cost)
        ''' 创建优化器。 '''
        self.optimizer = self.create_optimizer(optimist)
        pass

    ''' 根据配置创建损失函数。 '''
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]

    ''' 根据配置创建优化器。 '''
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]

    ''' 训练模型。 '''
    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                ''' 清除梯度，防止累积。 '''
                self.optimizer.zero_grad()
                ''' 前向传播，计算输出。 '''
                outputs = self.net(inputs)
                ''' 计算损失。 '''
                loss = self.cost(outputs, labels)
                ''' 反向传播，计算梯度。 '''
                loss.backward()
                ''' 更新权重。 '''
                self.optimizer.step()
                running_loss += loss.item()

                if i % 100 == 0:
                    ''' 每100个批次打印一次损失。 '''
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print('Finished Training')

    ''' 评估模型。 '''
    def evaluate(self, test_loader):
        print('Evaluating...')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                ''' 前向传播，计算输出。 '''
                outputs = self.net(inputs)
                ''' 获取预测结果。 '''
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        ''' 打印准确率。 '''
        print('Accuracy: %d %%' % (100 * correct / total))

''' 加载MNIST数据集。 '''
def mnist_load_data():
    ''' 定义数据预处理步骤，将图像转换为张量并归一化。 '''
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0, ], [1, ])])

    ''' 加载训练集，设置数据路径、是否为训练集、是否下载、数据预处理方式。 '''
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    ''' 创建训练集的数据加载器，设置批量大小、是否打乱顺序、工作线程数。 '''
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)

    ''' 加载测试集，设置数据路径、是否为训练集、是否下载、数据预处理方式。 '''
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    ''' 创建测试集的数据加载器，设置批量大小、是否打乱顺序、工作线程数。 '''
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True, num_workers=0)

    return train_loader, test_loader

''' 主函数，用于训练和评估模型。 '''
if __name__ == '__main__':
    ''' 实例化网络。 '''
    net = MnistNet()
    ''' 实例化模型，指定网络、损失函数和优化器。 '''
    model = Model(net, cost='CROSS_ENTROPY', optimist='RMSP')
    ''' 加载训练集和测试集。 '''
    train_loader, test_loader = mnist_load_data()
    ''' 训练模型。 '''
    model.train(train_loader)
    ''' 评估模型。 '''
    model.evaluate(test_loader)
