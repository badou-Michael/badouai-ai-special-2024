import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# 定义网络结构
class Model:
    # 初始化网络结构
    def __init__(self, net, cost, optimizer):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimizer)

    # 定义损失函数
    def create_cost(self, cost_name):
        create_cost_dict = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return create_cost_dict[cost_name]

    # 定义训练器
    def create_optimizer(self, optimizer_name, **rests):
        create_optimizer_dict = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return create_optimizer_dict[optimizer_name]

    # 定义训练过程
    def train(self, train_loader, epochs):
        for epoch in range(epochs):
            # 初始化损失值
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # 获取输入和标签
                inputs, labels = data
                # 梯度清零
                self.optimizer.zero_grad()
                # 传入输入 得出输出结果
                outputs = self.net(inputs)
                # 根据输出结果和标签计算损失值
                loss = self.cost(outputs, labels)
                # 反向传播
                loss.backward()
                # 更新参数
                self.optimizer.step()
                # 记录损失值
                running_loss += loss.item()
                if i % 2000 == 0:  # 每2000个batch打印一次损失值
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))

    # 定义测试过程
    def test(self, test_loader):
        # 初始化正确率和总数
        correct = 0
        total = 0
        with torch.no_grad():  # 关闭梯度计算
            # 遍历测试集
            for data in test_loader:
                images, labels = data  # 获取输入和标签
                # 传入输入 得出输出结果
                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)  # 计算预测类别
                total += labels.size(0)  # 累计总数
                # 计算正确率
                correct += (predicted == labels).sum().item()
                # 打印测试结果
                print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


# 加载数据集
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


# 定义网络结构
class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    # 定义向前传播过程
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.softmax(self.fc3(x))
        return x


# 主函数
if __name__ == '__main__':
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader, 10)
    model.test(test_loader)
