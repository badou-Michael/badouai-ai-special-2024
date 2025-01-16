import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 数据增强
def augment_data(x):
    return x + 0.1 * torch.randn_like(x)

# 模拟数据集
train_X = torch.rand(100, 784)  # 训练数据（100条样本）
train_Y = torch.randint(0, 10, (100,))  # 训练标签
train_X = augment_data(train_X)  # 数据增强

test_X = torch.rand(20, 784)  # 测试数据（20条样本）
test_Y = torch.randint(0, 10, (20,))  # 测试标签

# 数据加载器
train_dataset = TensorDataset(train_X, train_Y)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

test_dataset = TensorDataset(test_X, test_Y)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# 定义神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 200)  # 全连接层1
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(200, 10)  # 输出层

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练模型
for epoch in range(10):
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()  # 清除梯度
        outputs = model(batch_X)  # 前向传播
        loss = criterion(outputs, batch_Y)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for batch_X, batch_Y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        total += batch_Y.size(0)
        correct += (predicted == batch_Y).sum().item()

print(f"测试集准确率：{100 * correct / total:.2f}%")
