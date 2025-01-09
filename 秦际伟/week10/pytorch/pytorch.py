import torch                                    # 导入 PyTorch 库
import torch.nn as nn                           # 导入神经网络模块
import torch.optim as optim                     # 导入优化器模块
import torch.nn.functional as F                 # 导入函数模块
import torchvision                              # 导入 torchvision 库
import torchvision.transforms as transforms     # 导入图像变换模块


class Model:
    def __init__(self, net, cost, optimist):
        self.net = net                                       # 初始化神经网络模型
        self.cost = self.create_cost(cost)                   # 初始化损失函数
        self.optimizer = self.create_optimizer(optimist)     # 初始化优化器


    def create_cost(self, cost):    # 创建损失函数方法
        support_cost = {                                     # 支持的损失函数字典
            'mse': nn.MSELoss(),                             # 均方误差损失函数
            'cross_entropy': nn.CrossEntropyLoss()           # 交叉熵损失函数
        }
        return support_cost[cost]                            # 返回指定的损失函数


    # 创建优化器方法
    def create_optimizer(self, optimist, **rests):
        support_optimist = {                                                   # 支持的优化器字典
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),          # 随机梯度下降优化器
            'ADAM': optim.Adam(self.net.parameters(), lr=0.001, **rests),      # Adam 优化器
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)   # RMSprop 优化器
        }
        return support_optimist[optimist]

    # 训练函数方法
    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):                                     # 指定迭代的epoch数
            running_loss = 0.0                                           # 初始化损失值
            for i, data in enumerate(train_loader, 0):                   # 遍历训练数据
                inputs, labels = data                                    # 获取输入和标签

                self.optimizer.zero_grad()                               # 梯度清零

                outputs = self.net(inputs)                               # 前向传播
                loss = self.cost(outputs, labels)                        # 计算损失
                loss.backward()                                          # 反向传播
                self.optimizer.step()                                    # 更新参数
                running_loss += loss.item()                              # 累加损失
                if i % 100 == 0:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0                                   # 重置损失值

            # print('Finished Training')


    # 评估函数方法
    def evaluate(self, test_loader):
        print('Evaluating...')                                  # 开始评估
        correct = 0                                             # 初始化正确预测的数量
        total = 0                                               # 初始化总预测的数量
        with torch.no_grad():                                   # 关闭梯度计算，在评估时不计算梯度，节省内存和计算时间
            for data in test_loader:                            # 遍历测试数据
                images, labels = data                           # 获取输入和标签

                outputs = self.net(images)                      # 前向传播
                _, predicted = torch.max(outputs.data, dim=1)   # 获取预测结果
                total += labels.size(0)                         # 累加总预测的数量
                correct += (predicted == labels).sum().item()   # 累加正确预测的数量
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

def mnist_load_data():
    # 定义数据变换 (将图像转换为张量，并归一化)
    transform = transforms.Compose(  # 定义图像变换
        [transforms.ToTensor(),  # 将图像转换为张量
         transforms.Normalize([0, ], [1, ])])  # 归一化图像

    # 加载MNIST数据集
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 加载测试数据
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # 创建测试数据加载器
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    # 返回数据加载器
    return train_loader, test_loader


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()                               # 继承自 torch.nn.Module
        self.fc1 = torch.nn.Linear(28*28, 512)             # 第一个全连接层，输入为28*28，输出为512
        self.fc2 = torch.nn.Linear(512, 512)    # 第二个全连接层，输入为512，输出为512
        self.fc3 = torch.nn.Linear(512, 10)     # 第三个全连接层，输入为512，输出为10

    def forward(self, x):             # 前向传播
        x = x.view(-1, 28*28)                       # 将输入转换为二维张量，每个样本的数据被展平为一个一维向量
        x = F.relu(self.fc1(x))                     # 通过第一个全连接层，并使用ReLU激活函数
        x = F.relu(self.fc2(x))                     # 通过第二个全连接层，并使用ReLU激活函数
        x = F.softmax(self.fc3(x), dim=1)           # 通过第三个全连接层，并使用Softmax激活函数
        return x


if __name__ == '__main__':
    net = MnistNet()                                                    # 创建MnistNet模型实例
    model = Model(net, 'cross_entropy', 'RMSP')          # 创建Model实例,指定损失函数和优化器
    train_loader, test_loader = mnist_load_data()                      # 加载训练集和测试集的数据加载器
    model.train(train_loader)                                          # 训练模型
    model.evaluate(test_loader)                                        # 评估模型
    print('Finished Training')
