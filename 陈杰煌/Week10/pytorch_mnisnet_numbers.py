import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Model:
    def __init__(self, model, loss_name, optimizer_name, device):
        self.device = device
        self.model = model.to(device)  # 将模型迁移到设备
        self.loss_fn = self.create_loss_fn(loss_name)
        self.optimizer = self.create_optimizer(optimizer_name)

    def create_loss_fn(self, loss_name):
        supported_losses = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return supported_losses[loss_name]

    def create_optimizer(self, optimizer_name, **kwargs):
        supported_optimizers = {
            'SGD': optim.SGD(self.model.parameters(), lr=0.1, **kwargs),
            'ADAM': optim.Adam(self.model.parameters(), lr=0.01, **kwargs),
            'RMSP': optim.RMSprop(self.model.parameters(), lr=0.001, **kwargs)
        }
        return supported_optimizers[optimizer_name]

    def train(self, train_loader, epochs=3):
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # 将数据迁移到设备
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # 清零梯度
                self.optimizer.zero_grad()

                # 前向传播
                outputs = self.model(inputs)

                # 计算损失
                loss = self.loss_fn(outputs, labels)

                # 反向传播 + 参数更新
                loss.backward()
                self.optimizer.step()

                # 累积损失并打印
                running_loss += loss.item()
                if (batch_idx + 1) % 100 == 0:
                    print(f'[Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}] '
                          f'Loss: {running_loss / 100:.4f}')
                    running_loss = 0.0
        print('Training Completed')

    def evaluate(self, test_loader):
        """
        评估模型在测试集上的准确率
        :param test_loader: 测试数据加载器
        """
        print('Evaluating...')
        self.model.eval()  # 切换到评估模式
        total = 0
        correct = 0
        with torch.no_grad():  # 禁用梯度计算
            for inputs, labels in test_loader:
                # 将数据迁移到设备
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)  # 前向传播
                predicted = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')


def load_mnist_data(device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0], [1])
    ])

    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=True, num_workers=8)

    return train_loader, test_loader


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # 加载数据集
    train_loader, test_loader = load_mnist_data(device)

    # 初始化模型和包装类
    model = MnistNet()
    trainer = Model(model, 'CROSS_ENTROPY', 'RMSP', device)

    # 训练模型
    trainer.train(train_loader)

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        trainer.evaluate(test_loader)
