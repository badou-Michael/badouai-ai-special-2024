import torch
import torch.nn.functional as func
import torchvision
import torchvision.transforms as transforms

'''
用torch实现一个简单的FC网络
'''

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.W = torch.nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):  # 正向的过程，做wx+b
        x = x.mm(self.W)  # mm是matmul点乘
        if self.bias:
            x = x + self.bias.expand_as(x)
        return x


'''
用torch实现一个图像识别
'''

# 创建一个类，用来初始化模型，以及初步设置模型的框架
class MnistNet(torch.nn.Module):
    def __init__(self):  # init中放需要训练的层
        # super(MnistNet, self).__init__()
        super().__init__()  # 在Python3中不需要显式的指定类名和实例
        self.fc1 = torch.nn.Linear(28 * 28, 512)  # FC网络第一层计算
        self.fc2 = torch.nn.Linear(512, 512)  # FC网络第二层计算的
        self.fc3 = torch.nn.Linear(512, 10)  # FC网络输出层，转成10个特征张量了

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = func.relu(self.fc1(x))  # 将输入的张量代入到fc1中，然后再代入到relu激活函数中，完成一个从输入到第一个计算层的完整过程（计算wx+b，计算激活函数）
        x = func.relu(self.fc2(x))  # 同上，进入第二个计算层
        x = func.softmax(self.fc3(x), dim=1)  # 同上，区别就是最后这层用softmax作为激活函数，dim是指softmax在x的哪个维度上进行概率结果的判断，1就是在第二个维度
        return x


# 创建一个类，用来细化模型中使用的损失函数，以及实现正式的训练和测试
class Model:
    def __init__(self, net, cost, optimist):  # 计划传入网络、损失函数和一些其它参数
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)

    def create_cost(self, cost):  # 定义一个方法，用于选择损失函数
        support_cost = {
            'CROSS_ENTROPY': torch.nn.CrossEntropyLoss(),
            'MSE': torch.nn.MSELoss(),
        }
        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):  # 定义一个方法，用于选择模型的优化器，优化器实际作用不是特别大，但聊胜于无
        suport_optim = {
            'SGD': torch.optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'Adam': torch.optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': torch.optim.RMSprop(self.net.parameters(), lr=0.001, **rests),
        }
        return suport_optim[optimist]

    def train(self, train_loader, epochs):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):  # 用enumerate会在正常迭代取出需要的东西的同时，连带着把当时的索引也取出来，因此需要两个变量来接收
                inputs, labels = data[0].to(device), data[1].to(device)  # 默认数据集带过来的训练集和答案
                self.optimizer.zero_grad()
                '''
                为什么需要 zero_grad()
                在 PyTorch 中，梯度是自动累积的。
                这意味着如果你不手动清空梯度，每次执行 backward() 时，新的梯度会被累加到现有的梯度上，而不是覆盖它们。
                这可能会导致梯度值变得非常大，从而影响模型的训练过程，甚至导致数值不稳定或模型无法收敛。
                因此，在每次前向传播（forward pass）之后、反向传播之前，你需要调用 optimizer.zero_grad() 来将所有参数的梯度重置为零。
                这样可以确保每次计算的梯度都是基于当前批次的数据，而不是累积了之前批次的梯度。
                '''
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)  # 这里调用了在当时选择的损失函数，然后它需要两个参数，一个结果一个答案，得出损失数
                loss.backward()  # 上一步给交叉熵传参计算后，用得出的损失结果去进行反向传播，即backward
                self.optimizer.step()  # 这个是执行参数更新的重要一步，反向传播完成后，更新权重

                running_loss += loss.item()  # 使用item()方法可以只提取loss张量中的标量值，用于计算总损失值
                if i % 100 == 0:
                    print('epoch：{}，{:.2f}%，loss:{:.3f}'.format(
                        epoch + 1,
                        (i - 1) * 1.0 / len(train_loader),
                        running_loss / 100
                    ))
                    running_loss = 0.0

    def evaluate(self, test_loader):
        print('推理中')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('正确率：{}%'.format(100 * correct / total))

# 定义一个函数，用来对数据进行初始化
def mnist_load_data():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.,), (1,))
        ]
    )
    '''
    transforms.ToTensor()：
    这是一个非常常用的变换操作，它的主要功能是将输入的图像数据转换为 PyTorch 的张量格式。
    通常情况下，如果图像数据是以 Python Imaging Library (PIL) 图像格式或者其他类似格式存在的，ToTensor 操作会进行以下转换：
    将图像的维度顺序从 (height, width, channels，即HWC)（例如对于彩色图像，height 表示图像高度，width 表示图像宽度，channels 表示颜色通道数，常见的 RGB 图像通道数为 3）
    调整为 (channels, height, width，即CHW)，以符合 PyTorch 张量对于图像数据维度表示的惯例。
    同时，会将图像的像素值从取值范围（比如常见的 [0, 255] ，对于 8 位的图像表示）归一化到 [0, 1] 区间，使得像素值以浮点数的形式在这个新的范围内表示，便于后续的数值计算以及与神经网络模型的输入要求相匹配。
    所以概括来说，ToTensor做三件事
    1）改变维度顺序，当已经是CHW的则不再处理
    2）01归一化处理
    3）数据类型转换成float32，与2）同时进行
    
    transforms.Normalize((0.,), (1.,))：
    Normalize 操作主要用于对图像张量进行归一化处理，其目的是调整数据的分布，使得数据更符合模型训练的期望，有助于提高模型的收敛速度和性能。
    它接受两个参数，第一个参数是一个元组，表示每个通道的均值，第二个参数同样是一个元组，表示每个通道的标准差。
    在这个具体的例子中，传入的均值元组是 (0.,) ，标准差元组是 (1,)，
    意味着对图像张量的所有通道（如果是彩色图像就是对 RGB 三个通道一起，如果是灰度图像就这一个通道）进行归一化操作，将每个通道的数据减去均值 0 ，再除以标准差 1 。
    实际上，这样的操作相当于没有改变数据本身的值（因为减去 0 不影响数值，除以 1 也保持原数不变），
    不过在实际应用中，通常会根据具体数据集的统计特征来设置合适的均值和标准差，比如对于 MNIST 数据集会设置更贴合其数据分布的均值和标准差来进行真正有效的归一化操作。
    '''

    # 设置要执行的训练集的来源和一些参数
    trainset = torchvision.datasets.MNIST(
        root='./data',  # 读取数据的位置
        train=True,  # 是否是训练数据，是就加载训练集，否就加载测试集
        download=True,  # 是否需要下载
        transform=transform  # 定义的对图片预处理的操作，如果没有则可以不写，写了就得是调用的方式实现
    )

    # 同理，设置测试集
    testset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # 设置训练集的读取，以及训练时的参数
    trainloader = torch.utils.data.DataLoader(
        trainset,  # 使用的训练集
        batch_size=64,  # 每次读取训练集时的大小
        shuffle=True,  # 是否打乱顺序
        num_workers=2,  # 多进程可用的进程数，默认0，代表只用主进程
    )

    # 同理，设置测试的一些参数
    testloader = torch.utils.data.DataLoader(
        testset,  # 使用的训练集
        batch_size=64,  # 每次读取训练集时的大小
        shuffle=True,  # 是否打乱顺序
        num_workers=2,  # 多进程可用的进程数，默认0，代表只用主进程
    )

    return trainloader, testloader


if __name__ == '__main__':
    # net = Linear(10, 10)
    # x = net.forward
    # print('11', x)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = MnistNet()
    net.to(device)
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')  # 选择使用交叉熵作为反向传播的损失函数，RMSP作为优化项
    trainloader, testloader = mnist_load_data()  # 用刚刚的初始化数据的函数，将数据搞出来
    model.train(trainloader, 5)
    # model.evaluate
    model.evaluate(testloader)














