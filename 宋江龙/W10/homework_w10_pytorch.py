#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/12/6 00:15
@Author  : Mr.Long
"""
import os.path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class HomeworkW10Model(object):
    def __init__(self, network, cst, opti_mist):
        self.network = network
        self.cost = self.create_cost_w10(cst)
        self.optimizer = self.create_optimizer_w10(opti_mist)

    def create_cost_w10(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]

    def create_optimizer_w10(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.network.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.network.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.network.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]

    def train_w10(self, train_loader, epochs=3):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.network(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print('Finished Training')

    def evaluate_w10(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict
            for data in test_loader:
                images, labels = data
                outputs = self.network(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def mnist_load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0,], [1,])])
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True, num_workers=2)
    return train_loader, test_loader


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    # train for mnist
    net = MnistNet()
    model = HomeworkW10Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train_w10(train_loader)
    model.evaluate_w10(test_loader)

