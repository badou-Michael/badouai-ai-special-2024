import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T

class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        # 输入-隐藏
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        # 隐藏-隐藏,两个隐藏层
        self.fc2 = torch.nn.Linear(512, 512)
        # 输出变成0-9标签
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        # 图片二维数组变成一维输入层
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class Model:
    def __init__(self, net, cost, optimist):
        """
        初始化模型
        @param net: 神经网络结构
        @param cost: 损失函数
        @param optimist: 优化项
        """
        self.net = net
        self.cost = self.createCost(cost)
        self.optimizer = self.createOptimizer(optimist)
        pass

    def createCost(self, cost):
        supcost = {
            "CROSS_ENTROPY": torch.nn.CrossEntropyLoss(),  # 交叉熵
            "MSE": torch.nn.MSELoss()  # MSE
        }
        return supcost[cost]

    def createOptimizer(self, optimist, **rests):
        """
        优化算法
        @param optimist: SGD：
        @param rests:
        @return:
        """
        supoptimlist = {
            "SGD": optim.SGD(self.net.parameters(), lr=0.1, **rests),
            "ADAM": optim.Adam(self.net.parameters(), lr=0.01, **rests),
            "RMSP": optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return supoptimlist[optimist]

    def train(self,traindata,epoches=3):
        """
        训练
        @param traindata: 训练数据
        @param epoches: 训练迭代次数
        @return:
        """
        for epoch in range(epoches):
            running_loss = 0.0
            for i,data in enumerate(traindata,0):
                inputs,labels = data
                #梯度归零，即上次的计算梯度记录会被清空
                self.optimizer.zero_grad()
                #输出
                outputs = self.net(inputs)
                #损失函数
                loss = self.cost(outputs,labels)
                #反向传播计算得到每个参数的梯度值
                loss.backward()
                #通过梯度下降执行一步参数更新
                self.optimizer.step()
                #损失累加
                running_loss += loss.item()
                #每一代的损失打印
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(traindata), running_loss / 100))
                    running_loss = 0.0

    def evaluate(self,test_data):
        correct = 0
        total = 0
        with torch.no_grad():  #不计算梯度
            for data in test_data:
                images, labels = data
                outputs = self.net(images)
                #从大到小排序后取一个
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def mnist_load_data():
    transform = T.Compose(
        [T.ToTensor(), # 将图像转换为Tensor
         T.Normalize([0, ], [1, ])]) #进行标准化处理

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader

if __name__ == '__main__':
    # train for mnist
    #写好神经网络
    net = MnistNet()
    #定义模型
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    #加载训练数据与测试数据
    train_loader, test_loader = mnist_load_data()
    #模型训练
    model.train(train_loader,5)
    #测试预测
    model.evaluate(test_loader)