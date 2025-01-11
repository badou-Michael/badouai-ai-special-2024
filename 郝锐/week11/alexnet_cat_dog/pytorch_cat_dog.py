#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/12/14 13:27
# @Author: Gift
# @File  : pytorch_cat_dog.py 
# @IDE   : PyCharm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import csv

# 自定义猫狗数据集类
class CatDogDataset(Dataset):
    def __init__(self, data_dir, dataset_file, transform=None):
        """
        初始化猫狗数据集

        参数:
        data_dir: 图像数据所在的目录路径
        dataset_file: 标注文件（dataset.txt）的文件名
        transform: 可选的图像变换操作，用于数据预处理等
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 读取标注文件，解析出图像路径和对应的标签
        with open(os.path.join(data_dir, dataset_file), 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                self.image_paths.append(os.path.join(data_dir, row[0]))
                self.labels.append(int(row[1]))

    def __len__(self):
        """
        返回数据集的大小，即图像的数量
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的单个样本（图像和对应的标签）

        参数:
        idx: 样本的索引

        返回:
        image: 处理后的图像张量
        label: 对应的标签
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 构建AlexNet模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 第一个卷积层，96个11x11卷积核，步长为4，使用ReLU激活函数
            nn.Conv2D(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            # 最大池化层，池化窗口3x3，步长为2
            nn.MaxPool2D(kernel_size=3, stride=2),

            # 第二个卷积层，256个5x5卷积核，步长为1，填充为2，ReLU激活函数
            nn.Conv2D(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            # 最大池化层，池化窗口3x3，步长为2
            nn.MaxPool2D(kernel_size=3, stride=2),

            # 第三个卷积层，384个3x3卷积核，步长为1，填充为1，ReLU激活函数
            nn.Conv2D(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # 第四个卷积层，384个3x3卷积核，步长为1，填充为1，ReLU激活函数
            nn.Conv2D(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            # 第五个卷积层，256个3x3卷积核，步长为1，填充为1，ReLU激活函数
            nn.Conv2D(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 最大池化层，池化窗口3x3，步长为2
            nn.MaxPool2D(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2D((6, 6))
        self.classifier = nn.Sequential(
            # 第一个全连接层，将展平后的特征映射到4096维空间
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            # Dropout层，防止过拟合，训练时随机丢弃一半的神经元
            nn.Dropout(0.5),

            # 第二个全连接层，再次映射到4096维空间
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            # 输出层，输出节点数量等于类别数量（猫狗两类）
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 训练模型的函数
def train_model(model, device, train_loader, val_loader, epochs, learning_rate):
    """
    训练模型，并在验证集上进行评估

    参数:
    model: 要训练的神经网络模型
    device: 训练使用的设备（GPU或CPU）
    train_loader: 训练数据集的数据加载器
    val_loader: 验证数据集的数据加载器
    epochs: 训练的轮次
    learning_rate: 学习率

    返回:
    train_losses: 训练过程中的损失值列表
    val_losses: 验证过程中的损失值列表
    val_accuracies: 验证过程中的准确率值列表
    """
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        # 训练循环
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        # 计算训练集平均损失和准确率
        train_loss = running_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        # 验证模型
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        # 计算验证集平均损失和准确率
        val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

    return train_losses, val_losses, val_accuracies

# 简单验证模型的函数
def validate_model(model, device, test_loader):
    """
    使用测试集简单验证模型的准确率

    参数:
    model: 训练好的神经网络模型
    device: 验证使用的设备（GPU或CPU）
    test_loader: 测试数据集的数据加载器

    返回:
    accuracy: 模型在测试集上的准确率
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

# 数据预处理操作
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 假设图像和dataset.txt所在目录
data_dir = './data'
dataset_file = 'dataset.txt'

# 创建训练集数据集对象和数据加载器
train_dataset = CatDogDataset(data_dir, dataset_file, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 创建验证集数据集对象和数据加载器（这里简单划分一部分训练集作为验证集，你可根据实际调整）
val_dataset = CatDogDataset(data_dir, dataset_file, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 创建测试集数据集对象和数据加载器（同样可根据实际划分测试集，这里示例简单处理）
test_dataset = CatDogDataset(data_dir, dataset_file, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 检查是否有可用GPU，确定计算设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 构建AlexNet模型实例并将其移动到指定设备上
model = AlexNet().to(device)

# 训练轮次和学习率设置
epochs = 10
learning_rate = 0.001

# 训练模型，获取训练和验证过程中的损失和准确率信息
train_losses, val_losses, val_accuracies = train_model(model, device, train_loader, val_loader, epochs, learning_rate)

# 使用测试集验证训练好的模型
test_accuracy = validate_model(model, device, test_loader)
