'''
在pytorch中就只需要分三步：
1. 写好网络；
2. 编写数据的标签和路径索引；
3. 把数据送到网络。

pytorch 常用的有 torch.nn模块提供了创建神经网络的基础构件，线性层（linear layer）这些层都继承自Module类
通常对于可训练参数的层使用module
对应的在nn.functional模块中，提供这些层对应的函数实现。
对于不需要训练参数的层如softmax这些，可以使用functional中的函数。
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class Mnistnet(torch.nn.Module):
    """网络结构分的定义分为两层， 初始化定义网络每层节点数（权重矩阵），第二层处理数据，每层的激活函数计算"""
    def __init__(self):
        super(Mnistnet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        # 前两层的激活函数是relu，最后一层是softmax 生成概率值
        # self.fc1(x) 实现的是矩阵乘，输出的是矩阵乘的结果
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x


# 再来整体的模型，在网络的基础上还有反向传播-（损失函数，更新权重，优化项）
# 需要确定的是net，cost，optimist
class Model:
    def __init__(self, network, cost, optimist):
        self.network = network
        self.cost = self.creative_cost(cost)
        self.optimist = self.creative_optimist(optimist)

    def creative_cost(self, cost):
        support_cost = {
            'MSE': nn.MSELoss(),
            'CROSS_ENTROPY': nn.CrossEntropyLoss()
        }
        return support_cost[cost]

    def creative_optimist(self, optimist, **rests):
        # **传入一些优化方法额外的优化项数值，形成一个字典，在优化时会到这个字典里寻找
        # parameters是一个容器，返回的是一些可学习参数（权重和偏移量），
        # 在优化时需要知道那些参数是要被更新的，也就是优化的是谁
        support_optimist = {
            'SGD': optim.SGD(self.network.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.network.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.network.parameters(), lr=0.001, **rests)
        }
        return support_optimist[optimist]
    def train(self,train_loader, epoch=3):
        """训练过程（训练数据和训练周期），正向，反向，加优化"""
        for i in range(epoch):
            running_loss = 0.0
            for j, data in enumerate(train_loader, 0):
                batch_image_data, batch_label = data

                self.optimist.zero_grad()
                # 开始正向，损失，反向
                output = self.network(batch_image_data)
                loss = self.cost(output, batch_label)
                loss.backward()  # !!!!!!
                running_loss += loss.item()
                self.optimist.step()  #
                if j % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (i + 1, (j + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict
            for data in test_loader:
                images, labels = data

                outputs = self.network(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def mnist_load_data():
    """ 准备数据，数据一般放在torchvision   里面一般放数据和数据转换啥的 ， 还有需要数据加载器，
    用来规定训练时的batch大小"""
    # 定义一个数据转换方式，Compose是函数操作集合的意思，参数是一个列表，列表元素是一个个转换操作，
    # 当数据传入，会根据列表的的操作一步一步转化，转化成张量，标准化
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0, ], [1, ])])
    # 加载数据，得到准备好的数据集，是否是训练数据，是否下载，数据转换
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                          transform=transform)
    # 准备一个数据加载器，虽然已经把数据集准备好了，但我们需要用一个批次的数据来计算梯度，
    # 避免因为单张图的特征太独特，导致权重不断震荡，用一个批次的图可以有效的平均特性，使得具有普遍性的特征突出
    # 确定batch_size，打乱，以及计算资源
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    # 同样的加载训练数据
    testset = torchvision.datasets.MNIST(root='./', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader


if __name__ == '__main__':
    net = Mnistnet()
    # 定义网络 正向过程，最主要的是矩阵乘（线性乘）
    # 接下来定义损失函数和优化项，还有反向传播,模型类
    model = Model(net, 'CROSS_ENTROPY', 'SGD')
    # 模型整体弄好了，然后准备数据
    train_loader1, testl_oader1 = mnist_load_data()
    model.train(train_loader1)
    model.evaluate(testl_oader1)
