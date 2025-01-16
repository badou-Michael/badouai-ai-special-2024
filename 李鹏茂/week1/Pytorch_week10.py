import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__(self, net, cost, optimist, lr=0.01):
        # 将模型移到GPU或者CPU上
        self.net = net.to(device)
        # 创建损失函数
        self.cost = self.create_cost(cost)
        # 创建优化器
        self.optimizer = self.create_optimizer(optimist, lr)

    def create_cost(self, cost):
        # 根据传入的损失函数类型创建相应的损失函数
        if cost == 'CROSS_ENTROPY':
            return nn.CrossEntropyLoss()  # 分类问题常用的交叉熵损失函数
        elif cost == 'MSE':
            return nn.MSELoss()  # 回归问题常用的均方误差损失函数
        else:
            raise ValueError("Unsupported cost function: {}".format(cost))

    def create_optimizer(self, optimist, lr):
        # 根据传入的优化器类型创建相应的优化器
        if optimist == 'SGD':
            return optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)  # 使用SGD优化器，并加上momentum（动量）
        elif optimist == 'ADAM':
            return optim.Adam(self.net.parameters(), lr=lr)  # 使用Adam优化器
        elif optimist == 'RMSP':
            return optim.RMSprop(self.net.parameters(), lr=lr)  # 使用RMSprop优化器
        else:
            raise ValueError("Unsupported optimizer: {}".format(optimist))

    def train(self, train_loader, epoches=3):
        # 训练过程，指定训练的轮数
        for epoch in range(epoches):
            running_loss = 0.0
            # 遍历数据加载器中的数据
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                # 将数据移到GPU或CPU
                inputs, labels = inputs.to(device), labels.to(device)

                # 将梯度清零
                self.optimizer.zero_grad()

                # 前向传播：通过网络获取预测结果
                outputs = self.net(inputs)
                # 计算损失
                loss = self.cost(outputs, labels)
                # 反向传播：计算梯度
                loss.backward()
                # 更新模型参数
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:  # 每100个batch输出一次训练进度和损失
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        # 评估过程
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # 在评估过程中不需要计算梯度
            # 遍历测试集
            for data in test_loader:
                images, labels = data
                # 将数据移到GPU或CPU
                images, labels = images.to(device), labels.to(device)

                # 前向传播：获取网络的输出
                outputs = self.net(images)
                # 使用torch.argmax()获取最大值所在的索引，即分类结果
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)  # 总的样本数量
                correct += (predicted == labels).sum().item()  # 统计正确的分类数量

        # 输出准确率
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# 数据加载函数，返回训练集和测试集的数据加载器
def mnist_load_data():
    transform = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转图片，增加数据多样性
        transforms.RandomAffine(10, translate=(0.1, 0.1)),  # 随机仿射变换（包括平移）
        transforms.ToTensor(),  # 将图片转换为Tensor格式
        transforms.Normalize([0.], [1.])  # 对图片进行归一化处理
    ])

    # 加载训练集
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    # 加载测试集
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    return trainloader, testloader

# 定义一个简单的神经网络模型，用于MNIST分类
class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)  # 第一层全连接，输入28*28，输出512
        self.fc2 = torch.nn.Linear(512, 512)  # 第二层全连接，输入512，输出512
        self.fc3 = torch.nn.Linear(512, 10)  # 第三层全连接，输入512，输出10（10个类别）

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 将输入图片展平，变成一维向量
        x = F.relu(self.fc1(x))  # 第一层激活函数
        x = F.relu(self.fc2(x))  # 第二层激活函数
        x = self.fc3(x)  # 第三层输出，不再应用softmax，因为交叉熵损失已经内置softmax
        return x

# 主程序，开始训练和评估模型
if __name__ == '__main__':
    # 初始化网络和模型
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP', lr=0.001)  # 使用RMSprop优化器，学习率为0.001
    # 加载数据
    train_loader, test_loader = mnist_load_data()
    # 训练模型
    model.train(train_loader, epoches=3)
    # 评估模型
    model.evaluate(test_loader)
