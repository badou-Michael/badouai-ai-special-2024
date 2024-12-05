import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Model:
    def __init__(self, net, cost, optimist):
        # net：神经网络模型
        # cost：损失函数的类型
        # optimist：优化器的类型
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    # 根据传入的损失函数类型（交叉熵或均方误差），返回相应的损失函数对象
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    # 根据传入的优化器类型（SGD、Adam、RMSprop），返回相应的优化器对象
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                # 在反向传播之前，将模型参数的梯度归零
                self.optimizer.zero_grad()

                # forward + backward + optimize
                # 计算输入inputs的输出
                outputs = self.net(inputs)
                # 计算模型输出outputs和真实标签labels之间的损失
                loss = self.cost(outputs, labels)
                # 执行反向传播，计算损失关于模型参数的梯度
                loss.backward()
                # 根据计算出的梯度更新模型参数
                self.optimizer.step()

                # 当前批次的损失加到累计损失中
                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        # 预测正确的样本数量
        correct = 0
        # 测试集中的总样本数量
        total = 0
        # 禁用梯度计算
        with torch.no_grad():  # no grad when test and predict
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def mnist_load_data():
    # 定义了一个预处理流水线transform
    # transforms.ToTensor()：将PIL图像或NumPy数组转换为torch.FloatTensor，并且将图像的像素值从[0, 255]缩放到[0.0, 1.0]
    # transforms.Normalize([0, ], [1, ])：对图像数据进行标准化处理，这里将每个通道的均值标准化为0，标准差标准化为1。由于MNIST是灰度图像，所以均值和标准差都是针对单个通道
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0, ], [1, ])])

    # root：指定数据集的存储路径。
    # train：设置为True表示加载训练集。
    # download：设置为True表示如果数据集不存在则下载数据集。
    # transform：应用之前定义的预处理流水线。
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    # trainset：训练集数据集。
    # batch_size=32：设置每个批次的样本数量为32。
    # shuffle=True：设置为True表示在每个epoch开始时打乱数据。
    # num_workers=2：设置为2表示使用2个进程来加载数据，这可以加快数据加载速度
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
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
