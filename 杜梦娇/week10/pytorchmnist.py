import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#定义模型
class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass
    #定义损失函数选择函数
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    #定义优化器选择函数
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, device, train_loader, epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.net(data)
            loss = self.cost(output, target)
            loss.backward()
            self.optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    def test(self, device, test_loader):
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.net(data)
                test_loss += self.cost(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print(
            f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


#定义网络结构
class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

#定义正向传播及激活函数
    def forward(self, x):
        x = x.view(-1, 28*28)  # 展平图片
        x = torch.relu(self.fc1(x))  # 第一个隐藏层
        x = torch.relu(self.fc2(x))  # 第二个隐藏层
        x = self.fc3(x)           # 输出层
        return torch.softmax(x, dim=1)

# 定义转换操作，将图片转换为张量并归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def mnistdataloader():
    # 下载并加载训练数据
    trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    # 下载并加载测试数据
    testset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    return trainloader, testloader



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = myNet().to(device)
    model = Model(model, 'CROSS_ENTROPY', 'SGD')
    trainloader, testloader = mnistdataloader()
    for epoch in range(6):
        model.train(device, trainloader, epoch)
    model.test(device, testloader)
