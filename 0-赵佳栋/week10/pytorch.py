#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：pytorch.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/12/05 12:29
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


# 定义一个模型类
class Model:
    def __init__(self, net, cost, optimist):
        # 存储传入的神经网络
        self.net = net
        # 创建损失函数
        self.cost = self.create_cost(cost)
        # 创建优化器
        self.optimizer = self.create_optimizer(optimist)
        pass

    def create_cost(self, cost):
        # 支持的损失函数映射字典
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),  # 交叉熵损失函数，常用于分类任务
            'MSE': nn.MSELoss()  # 均方误差损失函数，常用于回归任务
        }
        # 根据传入的 cost 名称从字典中获取对应的损失函数
        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        # 支持的优化器映射字典
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),  # 随机梯度下降优化器，学习率为 0.1
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),  # Adam 优化器，学习率为 0.01
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)  # RMSprop 优化器，学习率为 0.001
        }
        # 根据传入的 optimist 名称从字典中获取对应的优化器
        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        # 开始训练，迭代多个 epoch
        for epoch in range(epoches):
            # 用于存储当前 epoch 的累计损失
            running_loss = 0.0
            # 遍历训练数据加载器
            for i, data in enumerate(train_loader, 0):
                # 获取输入数据和对应的标签
                inputs, labels = data
                # 清空优化器的梯度
                self.optimizer.zero_grad()
                # 前向传播，将输入数据传入神经网络得到输出
                outputs = self.net(inputs)
                # 计算损失，将输出和标签传入损失函数
                loss = self.cost(outputs, labels)
                # 反向传播，计算梯度
                loss.backward()
                # 根据梯度更新网络参数
                self.optimizer.step()
                # 累计当前批次的损失
                running_loss += loss.item()
                # 每 100 个批次打印一次训练进度和平均损失
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    # 重置累计损失
                    running_loss = 0.0
        # 训练完成打印信息
        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating...')
        # 正确预测的样本数量
        correct = 0
        # 总样本数量
        total = 0
        # 不计算梯度，因为在测试和预测阶段不需要更新参数
        with torch.no_grad():
            # 遍历测试数据加载器
            for data in test_loader:
                # 获取测试数据和对应的标签
                images, labels = data
                # 将测试数据传入神经网络进行前向传播得到输出
                outputs = self.net(images)
                # 获取预测结果，即输出中概率最大的类别索引
                predicted = torch.argmax(outputs, 1)
                # 统计总样本数量
                total += labels.size(0)
                # 统计正确预测的样本数量
                correct += (predicted == labels).sum().item()
        # 计算并打印测试集上的准确率
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


# 加载 MNIST 数据集的函数
def mnist_load_data():
    # 数据预处理，将图像转换为张量并归一化
    transform = transforms.Compose(
        [transforms.ToTensor(),  # 将图像转换为张量
         transforms.Normalize([0,], [1,])  # 归一化，均值为 0，标准差为 1
         ])
    # 加载 MNIST 训练集
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                    download=True, transform=transform)
    # 创建训练数据加载器，设置批处理大小为 32，打乱数据，使用 2 个工作线程
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                        shuffle=True, num_workers=2)
    # 加载 MNIST 测试集
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                   download=True, transform=transform)
    # 创建测试数据加载器，设置批处理大小为 32，打乱数据，使用 2 个工作线程
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader


# 定义 MNIST 网络类
class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        # 第一个全连接层，输入维度为 28*28，输出维度为 512
        self.fc1 = torch.nn.Linear(28*28, 512)
        # 第二个全连接层，输入维度为 512，输出维度为 512
        self.fc2 = torch.nn.Linear(512, 512)
        # 第三个全连接层，输入维度为 512，输出维度为 10
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        # 将输入的张量展平为一维向量，形状为 (batch_size, 28*28)
        x = x.view(-1, 28*28)
        # 经过第一个全连接层，并使用 ReLU 激活函数
        x = F.relu(self.fc1(x))
        # 经过第二个全连接层，并使用 ReLU 激活函数
        x = F.relu(self.fc2(x) )
        # 经过第三个全连接层，并使用 softmax 激活函数，将输出转换为概率分布
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    # 实例化 MnistNet 网络
    net = MnistNet()
    # 实例化 Model 类，使用 CROSS_ENTROPY 损失函数和 RMSP 优化器
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    # 加载 MNIST 数据集
    train_loader, test_loader = mnist_load_data()
    # 训练模型
    model.train(train_loader)
    # 评估模型
    model.evaluate(test_loader)