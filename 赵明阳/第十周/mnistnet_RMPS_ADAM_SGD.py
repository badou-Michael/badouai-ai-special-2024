import torch
#导入 torch 库，它是 PyTorch 的核心库，提供了张量操作、自动求导等功能。
import torch.nn as nn
#torch.nn 用于构建神经网络模块，如定义线性层、损失函数等。
import torch.optim as optim
#torch.optim 包含了各种优化算法，用于更新神经网络的参数。
import torch.nn.functional as F
#torch.nn.functional 提供了一些常用的神经网络操作函数（如激活函数等），这里以函数调用的形式使用，与直接在 nn 模块中定义层有所不同。
import torchvision
#torchvision 是用于处理图像数据的库，提供了常见的数据集（如 MNIST）以及图像变换等功能。
import torchvision.transforms as transforms
#torchvision.transforms 用于定义对图像数据进行的预处理操作，例如归一化、转换为张量等。

class Model:
    #__init__ 方法：这个类的初始化方法接收三个参数，net 是要训练的神经网络模型实例，cost 用于指定损失函数类型，optimist 用于指
    # 定优化器类型。在初始化函数中，分别调用 create_cost 和 create_optimizer 方法来创建对应的损失函数和优化器对象。
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass
    #create_cost 方法：定义了一个字典 support_cost，将字符串表示的损失函数名称（如 CROSS_ENTROPY 对应交叉熵损失，MSE 对应均方
    # 误差损失）映射到相应的 PyTorch 损失函数类实例。根据传入的 cost 参数，从字典中获取对应的损失函数对象并返回，用于后续计算
    # 模型预测结果与真实标签之间的误差。
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]
    #create_optimizer 方法：类似地，定义了一个字典 support_optim，将不同的优化器名称（如 SGD 表示随机梯度下降，ADAM 是一种自
    # 适应学习率的优化算法，RMSP 即 RMSProp 优化算法）映射到对应的 PyTorch 优化器类实例，并传入相应的默认学习率以及其他可能的
    # 参数（通过 **rests 接收额外参数）。根据传入的 optimist 参数，返回对应的优化器对象，用于更新神经网络的参数。
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]
    #这个方法实现了模型的训练过程，接受训练数据加载器 train_loader 和训练轮数 epoches（默认值为 3）作为参数。
    #在每一轮（epoch）训练中，首先初始化一个变量 running_loss 用于累计当前轮次的损失值。然后遍历训练数据加载器中的每一个批次数据，
    #对于每个批次：
    #
    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data #从数据中分离出输入图像数据 inputs 和对应的真实标签 labels。

                self.optimizer.zero_grad() #调用优化器的 zero_grad 方法，将之前一轮的梯度清零，避免梯度累积。

                # forward + backward + optimize
                #进行前向传播，通过 self.net(inputs) 得到模型的输出 outputs，再使用之前创建的损失函数 self.cost 计算预测输出
                # 与真实标签之间的损失值 loss。
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                #调用 loss.backward() 进行反向传播，计算梯度。
                loss.backward()
                #最后调用 self.optimizer.step() 根据计算得到的梯度来更新模型的参数。
                self.optimizer.step()

                running_loss += loss.item()
                #在每处理 100 个批次数据后，打印当前轮次的训练进度以及平均损失值，并且将 running_loss 清零重新累计。
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    #该方法用于评估训练好的模型在测试集上的性能。接收测试数据加载器test_loader作为参数。
    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        #在 torch.no_grad() 上下文环境下（表示不需要计算梯度，因为在测试阶段不需要进行反向传播和参数更新），遍历测试数据加载器
        # 中的每一个批次数据：
        with torch.no_grad():  # no grad when test and predict
            for data in test_loader:
                images, labels = data #从数据中分离出图像数据images和对应的真实标签labels。
                outputs = self.net(images)#通过模型得到输出 outputs，然后使用 torch.argmax 函数获取预测的类别（找到输出中
                                          # 概率最大的类别索引）。
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)#统计总样本数 total（通过 labels.size(0) 获取当前批次的样本数量并累加）以及预测正确
                                       # 的样本数 correct（通过比较预测结果和真实标签是否一致，并累加一致的数量）。
                correct += (predicted == labels).sum().item()
        #最后计算并打印模型在测试集上的准确率，以百分比的形式展示。
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
#这个函数用于加载 MNIST 数据集。首先定义了一个图像变换操作 transform，它由两个步骤组成：
def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])
        #transforms.ToTensor() 将图像数据转换为 PyTorch 的张量类型，同时将像素值范围从 [0, 255] 转换到 [0, 1]。
        #transforms.Normalize([0,], [1,]) 对数据进行归一化处理，这里以均值为 0、标准差为 1 进行归一化（MNIST
        # 数据本身比较简单，这样简单归一化也可行）。
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    #然后分别创建训练集 trainset 和测试集 testset，通过 torchvision.datasets.MNIST 类来获取 MNIST 数据集，设置好数据集的根
    # 目录（root='./data'，表示当前目录下的 data 文件夹，会自动下载数据集到该目录如果不存在的话）、是否为训练集（train=True
    # 表示训练集，train=False 表示测试集）以及应用之前定义的图像变换操作。
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    #最后使用 torch.utils.data.DataLoader 为训练集和测试集分别创建数据加载器 trainloader 和 testloader，设置了批次大小
    # （batch_size=32，即每次取 32 个样本作为一个批次进行训练或测试）、是否打乱数据顺序（shuffle=True）以及使用的工作进程数
    # （num_workers=2，用于加速数据加载过程，利用多进程并行读取数据），并返回这两个数据加载器。

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
    return trainloader, testloader

class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)
    #__init__ 方法：继承自 torch.nn.Module 类，在初始化函数中定义了三个全连接层（线性层）。self.fc1 是将输入的 28×28 维度的
    # 图像数据（展平后）映射到 512 维，self.fc2 也是 512 维的中间层，self.fc3 则将中间层的 512 维数据映射到 10 维，对应 MNIST
    # 数据集中的 10 个数字类别。
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
    #forward 方法：定义了数据在网络中的前向传播路径。首先使用 x.view(-1, 28*28) 将输入的图像张量（可能是包含批次维度的多维张量）
    # 展平为二维张量（批次大小 × 特征数量，这里特征数量就是 28×28 个像素值）。然后依次通过三个全连接层，并在每层之后应用激活函数，
    # 前两层使用 F.relu（ReLU 激活函数，用于引入非线性特性），最后一层使用 F.softmax 函数将输出转换为表示每个类别概率的分布
    # （在维度 dim=1 上进行，也就是对每个样本的输出进行概率归一化），最后返回处理后的输出结果。
if __name__ == '__main__':
    # train for mnist
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'SGD')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
#在 if __name__ == '__main__' 语句块中，首先创建了 MnistNet 类的实例 net，作为要训练的神经网络模型。
#然后创建 Model 类的实例 model，传入创建好的神经网络模型、指定的损失函数类型（CROSS_ENTROPY，即交叉熵损失）以及优化器类型
# （RMSP，即 RMSProp 优化器,Accuracy of the network on the test images: 95 %)）。
#(ADAM,Accuracy of the network on the test images: 18 %)
#(SGD,Accuracy of the network on the test images: 93 %)
#通过调用 mnist_load_data 函数获取训练集和测试集的数据加载器 train_loader 和 test_loader。
#接着调用 model.train(train_loader) 对模型进行训练，训练完成后再调用 model.evaluate(test_loader) 对训练好的模型在测试集上进行
# 性能评估，输出模型的准确率。
