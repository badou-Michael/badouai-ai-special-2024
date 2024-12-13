#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/12/7 10:25
# @Author: Gift
# @File  : torch-_mnist.py 
# @IDE   : PyCharm
import torchvision
import torch
def mnist_data_load():
    #数据预处理
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), #将数据转为tensor
        torchvision.transforms.Normalize((0,), (1,)) #数据归一化均值0.5，方差0.5
    ])
    #加载数据集
    #存放到当前目录下的data目录，下载的是train数据集，使用transform进行数据预处理，如果数据不存在则下载
    train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    #加载数据集，batch_size为64，shuffle为True，打乱数据,使用俩个进程来处理数据
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True,num_workers=2)
    #加载测试数据集train=False标识下载的是测试集
    test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True,num_workers=2)
    return train_loader,test_loader
#定义神经网络
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512) #输入层到隐藏层
        self.fc2 = torch.nn.Linear(512, 512) #隐藏层到隐藏层
        self.fc3 = torch.nn.Linear(512, 10) #隐藏层到输出层
    def forward(self, x):
        x = x.view(-1, 28*28) #将输入数据展平
        x = torch.relu(self.fc1(x)) #激活函数
        x = torch.relu(self.fc2(x)) #激活函数
        x = torch.softmax(self.fc3(x), dim=1) #输出层
        return x
#定义训练模型
class Model():
    def __init__(self, net, loss, optimizer):
        self.net = net
        self.loss = self.cal_loss(loss)
        self.optimizer = self.cal_optimizer(optimizer)
        pass
    def cal_loss(self, loss):
        supported_loss = {
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss(),
            'MSELoss': torch.nn.MSELoss()
        }
        return supported_loss[loss]
    def cal_optimizer(self, optimizer,  **rests):
        supported_optimizer = {
            'SGD': torch.optim.SGD(self.net.parameters(), lr=0.01,  **rests),
            'Adam': torch.optim.Adam(self.net.parameters(), lr=0.001, **rests),
            'RMSP': torch.optim.RMSprop(self.net.parameters(), lr=0.001,**rests)
        }
        return supported_optimizer[optimizer]
    def train(self, train_loader, epochs=3):
        for epoch in range(epochs):
            running_loss = 0.0
            for i,data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad() #梯度清零,每次迭代前梯度清零
                #前向传播
                outputs = self.net(inputs)
                #计算损失
                loss = self.loss(outputs, labels)
                #反向传播
                loss.backward()
                #更新参数
                self.optimizer.step()
                #打印损失
                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' % (epoch+1, 100. * i / len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print('Finished Training')
    #评估模型
    def evaluate(self, test_loader):
        print('evaluating...')
        correct = 0
        total = 0
        with torch.no_grad(): #不需要计算梯度,在评估和预测的时候
            for data in test_loader:
                images, labels = data
                outputs = self.net(images)
                predict = torch.argmax(outputs, dim=1) #返回每一行的最大值的索引
                total += labels.size(0) #总样本数
                correct += (torch.sum(predict == labels)).item() #预测正确的样本数
        print('Accuracy of the network on the 10000 test images: %.2f %%' % (100 * correct / total))
if __name__ == '__main__':
    net = Net()
    model = Model(net, 'CrossEntropyLoss', 'RMSP')
    train_loader, test_loader = mnist_data_load()
    model.train(train_loader, epochs=5)
    model.evaluate(test_loader)
