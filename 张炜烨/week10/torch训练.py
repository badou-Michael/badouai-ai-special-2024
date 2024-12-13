import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Model:
    def __init__(self, net, loss_function, optimizer_name):
        """
        初始化模型，包括网络结构、损失函数和优化器。
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net.to(self.device)  # 将网络放置在 GPU 或 CPU 上
        self.loss_function = self.create_loss_function(loss_function)
        self.optimizer = self.create_optimizer(optimizer_name)
    
    def create_loss_function(self, loss_function_name):
        """
        创建损失函数。
        """
        loss_functions = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        if loss_function_name not in loss_functions:
            raise ValueError(f"Unsupported loss function: {loss_function_name}")
        return loss_functions[loss_function_name]

    def create_optimizer(self, optimizer_name):
        """
        创建优化器。
        """
        optimizers = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001)
        }
        if optimizer_name not in optimizers:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        return optimizers[optimizer_name]

    def train(self, train_loader, epochs=3):
        """
        训练模型。
        """
        self.net.train()  # 切换到训练模式
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)  # 数据放置在 GPU 或 CPU 上

                # 清零梯度
                self.optimizer.zero_grad()

                # 前向传播 + 计算损失 + 反向传播 + 权重更新
                outputs = self.net(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # 记录损失
                running_loss += loss.item()
                if batch_idx % 100 == 0:  # 每100个batch打印一次日志
                    print(f"[Epoch {epoch + 1}, {100 * (batch_idx + 1) / len(train_loader):.2f}%] Loss: {running_loss / 100:.3f}")
                    running_loss = 0.0

        print("Finished Training")

    def evaluate(self, test_loader):
        """
        测试模型，计算准确率。
        """
        self.net.eval()  # 切换到评估模式
        correct = 0
        total = 0

        with torch.no_grad():  # 测试时不需要梯度计算
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.net(inputs)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy of the network on the test images: {accuracy:.2f}%")
        return accuracy


def load_mnist_data():
    """
    加载 MNIST 数据集，并进行必要的预处理。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 将像素值归一化到[-1, 1]
    ])

    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=32, shuffle=True, num_workers=2
    )

    test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=32, shuffle=False, num_workers=2
    )

    return train_loader, test_loader


class MnistNet(nn.Module):
    """
    定义用于 MNIST 数据集的神经网络。
    """
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平输入图像
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)  # 使用 log_softmax
        return x


if __name__ == '__main__':
    # 初始化网络
    net = MnistNet()

    # 创建模型
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')

    # 加载数据
    train_loader, test_loader = load_mnist_data()

    # 训练
    model.train(train_loader, epochs=5)

    # 测试
    model.evaluate(test_loader)