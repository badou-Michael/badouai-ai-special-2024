import torch
import torch.nn as nn
import torch.optim as optim # 提供优化器（如 SGD 和 Adam）以更新模型参数
import torch.nn.functional as F # 提供函数式 API，支持激活函数、损失函数等的直接调用
import torchvision  # 用于加载和预处理图像数据集
import torchvision.transforms as transforms  # 提供图像数据增强和预处理的工具（如归一化、随机裁剪等）

'''
通常对于可训练参数的层使用 torch.nn.Module，
而对于不需要训练参数的层如softmax这些，可以使用 torch.nn.functional 中的函数
'''


'''
Model：
负责模型的训练、优化器和损失函数的配置，以及训练和评估流程。
它可以与不同的网络架构配合使用，而不局限于某一个具体的网络。

MnistNet：
负责定义神经网络的架构，包括层定义（如全连接层）和前向传播逻辑（forward 方法）。
它只专注于模型结构，与训练或评估的具体逻辑无关
'''

'''
常见优化算法：
GD (Gradient Descent, 梯度下降)
SGD (Stochastic Gradient Descent, 随机梯度下降)
Momentum (动量法)
RMSProp
Adam (Adaptive Moment Estimation)
'''

class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    # 定义计算损失函数 CROSS_ENTROPY：交叉熵  MSE：均值平方差
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            # **rests 表示接受任意数量的关键字参数（key=value 的形式），并将它们以字典的形式传递给函数或方法
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            # 初始化损失值
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)  # MnistNet的实例net
                loss = self.cost(outputs, labels)
                loss.backward()  # 计算损失函数对所有模型权重的梯度（即误差的反向传播） 梯度会存储在每个权重的 .grad 属性中
                self.optimizer.step()  # 根据计算出的梯度更新模型权重

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d ,%.2f%%] loss:%.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                # print(f'[Epoch {epoch + 1}, {i + 1}] Loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0.0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)  # 返回张量的第 0 维大小，即张量的长度
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images:%d %%' % (100 * correct / total))
        # print(f'Accuracy of the network on the test images: {100 * correct / total:.2f} %')


def mnist_load_data():
    transform = transforms.Compose(  # 定义数据增强和归一化处理
        [transforms.ToTensor(),  # 将 PIL 图像或 NumPy 数组转换为张量
         transforms.Normalize([0, ], [1, ])]  # 对图像进行归一化，均值=0，标准差=1
    )

    # 加载 MNIST 训练集，下载到 './data' 文件夹，应用 transform 进行预处理
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # 创建训练集数据加载器，批次大小为 32，随机打乱数据，使用 2 个工作线程加载数据
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    # 加载 MNIST 测试集
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 创建测试集数据加载器，批次大小为 32，随机打乱数据，使用 2 个工作线程加载数据
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    return trainloader, testloader


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # view 类似于 reshape，-1 表示自动推断 batch 的大小  对应二维张量形状 (1,28*28)
        x = F.relu(self.fc1(x))  # 在调用 self.fc1(x) 时已经内部实现了 WX+b
        x = F.relu(self.fc2(x))  # 等价于 self.fc2.__call__(x)
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
