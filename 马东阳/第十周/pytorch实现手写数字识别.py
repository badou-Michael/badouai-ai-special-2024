import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

'''
1. 数据准备
2. 模型构建
3. 模型训练与验证
4. 预测与测试
'''

class Model:
    def __init__(self, net, cost, optimist):  # 网络，损失，优化器
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    # 损失函数
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]
    # 优化器  **（双星号）：可变数量的关键字参数，在函数定义时，**rests可以接受可变数量的关键字参数，这些参数被封装为一个字典
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    # 训练集训练
    def train(self, train_loader, epoches=10):
        for epoch in range(epoches):    # 最外层循环
            running_loss = 0.0    # 初始损失
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)    # 正向结果
                loss = self.cost(outputs, labels)
                loss.backward()    # 反向
                self.optimizer.step()
                running_loss += loss.item()
                # 每张图训练一个损失
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training......')

    # 测试
    def evaluate(self, test_loader):
        print('Evaluating ......')
        correct = 0
        total = 0

        with torch.no_grad():  # 禁用梯度计算
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('测试数据准确率: %d %%' % (100 * correct / total))

def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])   # 标准化，归一化
    # torchvision.datasets是Pytorch自带的一个数据库，我们可以通过代码在线下载数据，这里使用的是torchvision.datasets中的MNIST数据集
    # batch_size:每个批次加载的数据量大小，batch_size=32 表示每次加载 32 个样本用于训练。
    # shuffle:是否对数据进行随机打乱，通常在训练集上设置为 True，在验证集或测试集上设置为 False
    # num_workers:用于数据加载的子进程数量。默认是 0，即使用主线程加载数据。设置为大于0的数值可以开启多进程（或多线程）加载数据，通常这会加快数据加载速度，尤其是在 I/O 操作较多的情况下。
    # 下载数据,在程序文件下生成一个data文件夹，将数据下载到文件夹内
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    # 训练集
    train_data = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    # 测试集
    test_data = torch.utils.data.DataLoader(testset, batch_size=64,shuffle=True, num_workers=2)
    return train_data, test_data


class Net(torch.nn.Module):
    def __init__(self):  # 初始化需要的层
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    # 计算构建
    def forward(self, x):
        x = x.view(-1, 28*28)   # 展开为1维
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

if __name__ == '__main__':
    net = Net()    # 网络
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()   # 数据
    model.train(train_loader)
    model.evaluate(test_loader)