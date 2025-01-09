import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Step 1: 加载和预处理数据
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为 Tensor，值在 [0, 1] 范围内
    transforms.Normalize((0.5,), (0.5,))  # 对数据进行归一化处理
])

# 下载 MNIST 数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 加载数据集
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True,num_workers=3)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False,num_workers=3)

# Step 2: 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 第一个全连接层，输入28*28=784，输出128
        self.fc2 = nn.Linear(128, 64)     # 第二个全连接层，输入128，输出64
        self.fc3 = nn.Linear(64, 10)      # 输出层，输入64，输出10（每个数字一个输出）

    def forward(self, x):
        x = x.view(-1, 28*28)  # 展平输入图像
        x = torch.relu(self.fc1(x))  # 使用 ReLU 激活函数
        x = torch.relu(self.fc2(x))  # 使用 ReLU 激活函数
        x = self.fc3(x)  # 输出层

        return x

# Step 3: 设置损失函数和优化器
model = SimpleNN()
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数,里面包含了softmax，因此x = self.fc3(x) 不需要写成x = torch.softmax(self.fc3(x), dim=1)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 使用SGD优化器

# Step 4: 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # 将输入和标签送入模型
        optimizer.zero_grad()  # 清空梯度
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()
        if i % 100 == 99:  # 每 100 个小批量输出一次损失
            print(f"Epoch {epoch+1}, Step {i+1}, Loss: {running_loss / 100:.3f}")
            running_loss = 0.0

print('Finished Training')

# Step 5: 评估模型
model.eval()  # 设置模型为评估模式
correct = 0
total = 0
with torch.no_grad():  # 评估时不计算梯度
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # 获取预测标签
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy on the test set: {accuracy:.2f}%')

# 可视化部分测试集的手写数字
data_iter = iter(test_loader)
images, labels = data_iter.next()
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(5):
    axes[i].imshow(images[i].squeeze(), cmap='gray')
    axes[i].set_title(f"Label: {labels[i].item()}")
    axes[i].axis('off')
plt.show()
