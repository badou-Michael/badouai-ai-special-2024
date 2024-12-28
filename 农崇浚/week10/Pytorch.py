import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 定义一个具有两个隐藏层的全连接网络
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层到第一个隐藏层，输入尺寸28x28=784，输出128个神经元
        self.fc2 = nn.Linear(128, 64)  # 第一个隐藏层到第二个隐藏层，128 -> 64
        self.fc3 = nn.Linear(64, 10)  # 第二个隐藏层到输出层，64 -> 10（对应MNIST的10个类别）

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 将输入的28x28图像展平为一个向量
        x = F.relu(self.fc1(x))  # 使用ReLU激活函数
        x = F.relu(self.fc2(x))  # 使用ReLU激活函数
        x = self.fc3(x)  # 输出层，直接使用线性输出（不加激活函数）
        return x


# 定义一个封装模型训练和评估过程的类
class Model:
    def __init__(self, net, cost, optimist):
        # 初始化模型，损失函数和优化器
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)

    # 创建损失函数
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),  # 交叉熵损失（用于分类任务）
            'MSE': nn.MSELoss()  # 均方误差损失（用于回归任务）
        }
        return support_cost[cost]

    # 创建优化器
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.001, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.01, **rests)
        }
        return support_optim[optimist]

    # 训练模型
    def train(self, train_loader, epochs=3):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                # 清零之前的梯度
                self.optimizer.zero_grad()

                # 前向传播 + 计算损失 + 反向传播 + 更新参数
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新参数

                running_loss += loss.item()
                if i % 100 == 0:  # 每100个批次输出一次损失
                    print(f'[Epoch {epoch + 1}, {i + 1}/{len(train_loader)}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        print('Finished Training')

    # 在测试集上评估模型性能
    def evaluate(self, test_loader):
        print('Evaluating model...')
        correct = 0
        total = 0
        with torch.no_grad():  # 禁用梯度计算，提高推理效率
            for data in test_loader:
                images, labels = data

                # 获取模型输出
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)  # 获取预测类别

                total += labels.size(0)
                correct += (predicted == labels).sum().item()  # 统计预测正确的样本数

        accuracy = 100 * correct / total  # 计算准确率
        print(f'Accuracy on test data: {accuracy:.2f}%')


# 加载MNIST数据集
def load_mnist_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]  # 数据标准化，将像素值调整到[-1, 1]范围
    )

    # 加载训练集
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # 加载测试集
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    return trainloader, testloader


# 主函数
if __name__ == '__main__':
    # 初始化模型，选择损失函数和优化器
    net = SimpleNN()
    model = Model(net, 'CROSS_ENTROPY', 'ADAM')  # 使用交叉熵损失和Adam优化器

    # 加载MNIST数据
    train_loader, test_loader = load_mnist_data()

    # 训练模型
    model.train(train_loader, epochs=5)

    # 评估模型在测试集上的表现
    model.evaluate(test_loader)
