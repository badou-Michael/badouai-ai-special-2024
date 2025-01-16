import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

'''
1、创建神经网络、损失函数、优化器
'''
class Model:
    def __init__(self, net, cost, optimist):
        self.net = net                                         # net:神经网络模型
        self.cost = self.create_cost(cost)                     # cost：损失函数类型
        self.optimizer = self.create_optimizer(optimist)       # optimist：优化器类型

    def create_cost(self, cost):                               # 根据传入的字符串选择损失函数
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),            # 交叉熵损失函数
            'MSE': nn.MSELoss()                                # 均方误差损失函数
        }
        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):                    # create_optimizer 根据传入的优化器名称 optimist 来创建相应的优化器
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),          # self.net 是定义的神经网络，parameters() 返回的是模型中所有 可学习的参数，如权重和偏置。
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),       # lr=0.1  学习率
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)     # 通过 **rests 传递的额外参数，可能进一步调整优化器的配置
        }
        return support_optim[optimist]

    '''
    2、训练过程
    train_loader：是一个批量加载训练数据的迭代器
    enumerate() 用于遍历一个可迭代对象（如列表、元组等），并且返回每个元素的 索引 和 元素本身；
    enumerate(iterable, start=0)：iterable 是你要遍历的对象，start=0 表示索引起始值从 0 开始。
    '''
    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0                               # 是用来累计每个批次的损失（loss）的值
            for i, data in enumerate(train_loader, 0):       # i 是批次的索引，表示当前批次的编号
                inputs, labels = data                        # data 当前批次的数据，包含了两个元素：inputs：输入数据（例如，图像数据),labels：标签数据（例如，图像对应的标签数字）
                self.optimizer.zero_grad()                   # 清除之前计算的梯度，防止梯度累积。
                outputs = self.net(inputs)                   # 前向传播
                loss = self.cost(outputs, labels)            # 计算插值
                loss.backward()                              # 反向传播
                self.optimizer.step()                        # 更新参数
                running_loss += loss.item()                  # 累加当前批次的损失值
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' % (                       # %d：格式化输出一个整数
                    epoch+1, 100*((i + 1) / len(train_loader)), running_loss / 100))        # %.2f%%：格式化输出一个浮动数（进度百分比），保留两位小数
                    running_loss = 0.0                                              # %.3f：格式化输出一个浮动数（损失），保留三位小数
        print('Finished Training')                           # 提示训练完成
    '''
    3、预测过程（测试过程）
    predicted == labels,这是一个逐元素比较操作,返回一个布尔型张量（True 或 False），形状与 predicted 和 labels 相同
    predicted 是 [1, 0, 2]，labels 是 [1, 1, 2]， predicted == labels 返回 tensor([True, False, True])
    (predicted == labels).sum() = tensor(2)
    item() 方法用于将张量中的单个值（标量）提取为 Python 的原生数值类型（如 int 或 float）
    (predicted == labels).sum().item() = 2
    correct 是一个累计计数器,从而累积正确预测的总数
    '''
    def evaluate(self, test_loader):                        # 选择测试数据
        print('Evaluating ...')                             # 提示开始
        correct = 0                                         # 正确样本初始为0
        total = 0                                           # 总样本数初始为0
        with torch.no_grad():                               # 不需要更新梯度
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)   # 返回 outputs 沿指定维度（dim）的最大值的索引,dim=1：表示在每个样本（行）上，选择每个类别得分最大的类别
                total += labels.size(0)                     # 所有测试样本数据
                correct += (predicted == labels).sum().item()    # 累积正确预测的总数

        print('Accuracy of the network on the test images: %d %%' % ((correct / total) * 100))

'''
4、传入数据（训练集、测试集）
transforms.Compose 是一个用于将多个数据预处理步骤按顺序组合在一起的函数
transforms.ToTensor() 将图像的像素值转换为张量，并进行归一化的处理
transforms.Normalize(mean, std) 会对图像的每个通道进行 标准化（均值、标准差）
torchvision.datasets.MNIST 是 torchvision 库中的一个预定义的数据集类，用于加载 MNIST 手写数字数据集
root='./data'：指定数据集存储的位置；train=True：指定加载训练集，False 则加载测试集；
download=True：如果 root 目录下没有数据集，会自动从互联网下载 MNIST 数据集
transform=transform：指定对数据集中的每一张图像应用的转换操作
torch.utils.data.DataLoader：用来将 trainset 数据集加载到 train_loader 中的，使用的是 DataLoader 类
DataLoader 是 PyTorch 中用于高效加载和批量处理数据的工具
shuffle=True：设置为 True 表示每个 epoch（训练周期）开始时，会对数据进行随机打乱
num_workers=2：这个参数控制用于加载数据的子进程数
'''
def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32,shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)

    return train_loader, test_loader

'''
5、MnistNet 定义模型架构 （前面的Model 类封装了训练过程，整合了模型、损失函数和优化器）
定义了一个名为 MnistNet 的神经网络模型，它继承自 PyTorch 的 torch.nn.Module 基类
所有的神经网络模型都需要继承 torch.nn.Module，这样就能方便地使用 PyTorch 提供的各种功能（如自动求导、优化器等）
'''
class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()                              # 调用父类 torch.nn.Module 的初始化方法
        self.fc1 = torch.nn.Linear(28*28, 512)             # 全连接层 fc1，接收大小为 28*28 的输入，输出维度为 512
        self.fc2 = torch.nn.Linear(512, 512)     # 全连接层 fc2
        self.fc3 = torch.nn.Linear(512, 10)      # 全连接层 fc3

    def forward(self, x):                             # forward 方法定义了前向传播的过程，即数据如何在网络中流动并产生输出
        x = x.view(-1, 28*28)                         # -1 表示自动计算这一维的大小（即批次的大小），28x28 的二维图像展平为一维向量，以便输入到全连接层中
        x = F.relu(self.fc1(x))                       # 通过第一个全连接层 fc1 并应用 ReLU 激活函数
        x = F.relu(self.fc2(x))                       # 通过第二个全连接层 fc2 并应用 ReLU 激活函数
        x = F.softmax(self.fc3(x), dim=1)             # 通过第三个全连接层 fc3，然后应用 Softmax 激活函数
        return x                                      # dim=1，表示沿着每一行进行 softmax（每个样本的 10 个类别的概率和为 1）
                                                      # x 是一个大小为 [batch_size, 10] 的张量，表示每个样本属于各个类别的概率

'''
6、开始训练
if __name__ == '__main__':
这行代码的作用是判断是否在主程序中运行。如果当前脚本是直接运行的，而不是作为模块导入，那么 __name__ 的值会是 '__main__'，此时就会执行以下代码
'''
if __name__ == '__main__':
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
