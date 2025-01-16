'''
torch: PyTorch 的核心模块，提供了张量操作、自动求导系统以及其他多维数组运算功能。
torch.nn: 提供构建神经网络所需的类和函数，包括层（Layer）、激活函数和损失函数等。
torch.optim: 提供各种优化算法，用于在训练过程中更新模型的权重。
torch.nn.functional: 提供了一系列的函数，这些函数是 PyTorch
构建神经网络时所需的各种操作和激活函数的函数式接口。
torchvision: 提供处理图像和视频的常用工具，包括数据集类、图像转换操作以及预训练模型等。
torchvision.transforms: 提供图像预处理和增强的各种变换操作。
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

class Model:
    # 初始化模型、损失函数和优化器
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass
    # 损失函数对象
    def create_cost(self,cost):
        support_cost ={
            'CROSS_ENTROPY':nn.CrossEntropyLoss(),
            'MSE' : nn.MSELoss()
        }
        return support_cost[cost]
    # 创建优化器对象
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD':optim.SGD(self.net.parameters(),lr = 0.1, **rests),
            'ADAM':optim.Adam(self.net.parameters(),lr = 0.01, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(),lr = 0.001, **rests)

        }
        return support_optim[optimist]

    def train(self, train_loader, epoches =3):
        for epoch in range(epoches):
            # 初始化一个变量来累计每个epoch的损失。
            running_loss = 0.0
            # 遍历train_loader中的所有数据批次
            for i , data in enumerate(train_loader, 0):
                inputs , labels = data
                self.optimizer.zero_grad()
                # 在反向传播前清除之前的梯度。
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                # 进行反向传播
                loss.backward()
                # 更新模型的权重。
                self.optimizer.step()
                # 将当前批次的损失添加到累计损失中
                running_loss += loss.item()
                if i % 100 == 0:
                    # 每处理100个批次，打印当前epoch的进度和平均损失
                    print('===epoch %d, %.2f%%  loss:%.3f==='
                          %(epoch +1 ,(i + 1)*1./len(train_loader),running_loss /100))
                    running_loss = 0.0
        print('训练完成')
    # 评估模型在测试集上面的性能
    def evaluate(self, test_loader):
        print('Evaluate......')
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('测试图片的正确率为：%d %%'%(100 * correct / total))
# 加载MNIST数据集，归一化处理
def mnist_load_data():
    #创建Compose实例进行数据预处理，
    # ToTensor进行图像转换，缩放像素值到0~1之间
    # Normalize进行数值的归一化处理，均值标准差为0，1
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0,],[1,])])
    # 加载训练集
    trainset = torchvision.datasets.MNIST(root = './task10_detail/data',train = True,
                                          download = True,transform = transform)
    # 创建了一个数据加载器，并设置为32个样本每批，打乱数据顺序，并使用2个工作进程来加载数据
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 32,
                                              shuffle = True, num_workers = 2)
    # 测试数据训练集
    testset = torchvision.datasets.MNIST(root = './task10_detail/data',train = False,
                                         download = True,transform = transform)
    # 与训练集的加载器保持一致
    testloader = torch.utils.data.DataLoader(testset, batch_size =32,shuffle = True,
                                             num_workers = 2)
    return trainloader, testloader

# 主程序，参数传入
class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 512)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim = 1)
        return x
if __name__ == '__main__':
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
