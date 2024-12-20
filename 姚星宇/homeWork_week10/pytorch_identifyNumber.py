import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

if torch.cuda.is_available():
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available.")

# 定义模块
class Model:
    # 初始化
    def __init__(self, net, cost, optimist, device='cpu'):
        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.cost = self.create_cost(cost).to(self.device)
        self.optimizer = self.create_optimizer(optimist)
    # 创建损失函数
    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]
    # 创建优化器
    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]
    # 训练
    def train(self, train_loader, epochs=3):
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:  
                    print(f'[epoch {epoch + 1}, {(i + 1)*100./len(train_loader):.2f}%] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        print('Finished Training')
    # 评估
    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  
            for data in test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')
# 加载数据集
def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)
    
    return trainloader, testloader

# 定义模型
class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # 展平输入图像
        x = F.relu(self.fc1(x))  # 第一层全连接 + ReLU 激活
        x = F.relu(self.fc2(x))  # 第二层全连接 + ReLU 激活
        x = self.fc3(x)  # 输出层，不应用 softmax
        return x

if __name__ == '__main__':
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化网络并创建模型实例
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP', device=device)

    # 加载数据
    train_loader, test_loader = mnist_load_data()

    # 确保模型和数据都在同一设备上
    model.train(train_loader, epochs=5)
    model.evaluate(test_loader)