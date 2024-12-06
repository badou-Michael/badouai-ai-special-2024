import torch.nn
from torch import nn, optim
import torch.nn.functional as torchfunc
import torchvision
import torchvision.transforms as transforms

class Model:
    def __init__(self, net, cost, optimizer):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimizer)

    def create_cost(self, cost):
        support_cost = {
            "CROSS_ENTROPY": nn.CrossEntropyLoss(),
            "MSE": nn.MSELoss()
        }
        return support_cost[cost]

    # **rests代表多个额外参数汇总成的字典
    def create_optimizer(self, optimizer, **rests):
        support_optimizer = {
            "SGD": optim.SGD(self.net.parameters(), lr = 0.1, **rests)
            "RMSP": optim.RMSprop(self.net.parameters(), lr = 0.01, **rests)
        }
        return support_optimizer[optimizer]

    def train(self, train_loader, epoches = 5):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                # 防止梯度累计，每次更新只使用当前批次的梯度
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0
        print("训练结束")

    def evalute(self, test_loader):
        correct = 0; total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data

                output = self.net(images)
                pridict_value = torch.argmax(output, dim=1)
                total += labels.size[0]
                correct += (pridict_value == labels).sum().item()
            print("网络的正确率为:", 100 * correct / total, "%%")

def mnist_load_data():
    transform = transforms.Compose(
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


class MinistNet(torch.nn.Module):
    def __init__(self):
        super(MinistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torchfunc.relu(self.fc1(x))
        x = torchfunc.relu(self.fc2(x))
        x = torchfunc.softmax(self.fc3(x), dim=1)
        return x

# 建模型
net = MinistNet()
model = Model(net, "CROSS_ENTROPY", "RMSP")
# 取数据
train_loader, test_loader = mnist_load_data()
# 喂数据
model.train(train_loader)
model.evalute(test_loader)