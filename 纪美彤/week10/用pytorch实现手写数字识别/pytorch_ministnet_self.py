import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.sparse import softmax


# torch.nn.Module提供了神经网络的基类，当实现神经网络时需要继承自此模块，并在初始化函
# 数中创建网络需要包含的层，并实现forward函数完成前向计算，网络的反向计算会由自动求
# 导机制处理。
# 通常将需要训练的层写在init函数中，将参数不需要训练的层在forward方法里调用对应的函数
# 来实现相应的层。
class MnistNet_self(torch.nn.Module):
    def __init__(self):
        super(MnistNet_self, self).__init__()

        # 需要训练的层
        self.full_connect_layer1 = torch.nn.Linear(28*28, 512)
        self.full_connect_layer2 = torch.nn.Linear(512, 512)
        self.full_connect_layer3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        # view修改tensor的shape
        x = x.view(-1, 28*28)
        # x进入full_connect_layer1后经过relu激活
        x = F.relu(self.full_connect_layer1(x))
        x = F.relu(self.full_connect_layer2(x))
        x = F.softmax(self.full_connect_layer3(x), dim = 1)
        return x

class Model_self:
    def __init__(self, net, loss_fn, optimizer):
        self.net = net
        self.loss_fn = self.create_cost(loss_fn)
        self.optimizer = self.create_optimizer(optimizer)
        pass

    def create_cost(self, loss_fn):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[loss_fn]

    def create_optimizer(self,optimizer, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimizer]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                # forward + backward + optimize
                # 正向传播
                outputs = self.net(inputs)
                # 反向传播
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                # optimize
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        # 计数
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
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0, ], [1, ])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader


if __name__ == '__main__':
    # 定义模型
    net = MnistNet_self()
    # 定义训练
    model = Model_self(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
