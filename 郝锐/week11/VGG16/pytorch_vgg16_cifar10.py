#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/12/14 12:13
# @Author: Gift
# @File  : pytorch_vgg16_cifar10.py 
# @IDE   : PyCharm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# 数据预处理与加载部分
# 定义数据预处理操作，用于训练集的数据增强和归一化等处理
transform_train = transforms.Compose([
    # 对图像进行随机裁剪，裁剪尺寸为32x32（CIFAR10图像原始大小），并在周围填充4个像素
    # 这是一种数据增强的方式，增加了数据的多样性，有助于模型更好地泛化
    transforms.RandomCrop(32, padding=4),
    # 以0.5的概率对图像进行随机水平翻转，进一步增加数据的多样性
    transforms.RandomHorizontalFlip(),
    # 将图像数据转换为PyTorch的张量格式，其值范围是[0, 1]，便于后续在模型中进行计算
    transforms.ToTensor(),
    # 对图像张量进行归一化操作，给定的均值和标准差是根据CIFAR10数据集的统计信息得出的
    # 这样可以使数据分布更符合模型训练的要求，有助于加快模型收敛速度和提高性能
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 定义测试集的数据预处理操作，相对训练集少了数据增强部分，因为测试时不需要改变数据本身的特征
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载训练集
# root参数指定数据集下载和保存的根目录，这里设置为当前目录下的'data'文件夹
# train=True表示加载训练集数据
# download=True表示如果数据集不存在则自动下载
# transform参数传入之前定义好的训练集预处理操作
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
# 使用DataLoader将训练集数据封装成可迭代的数据加载器，方便在训练过程中按批次获取数据
# batch_size=64表示每个批次包含64个样本
# shuffle=True表示在每个epoch开始前打乱数据集的顺序，使模型在不同顺序的数据上训练，增强泛化能力
# num_workers=2表示使用2个子进程来并行加载数据，加快数据加载速度（具体数值可根据实际硬件情况调整）
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

# 加载测试集，参数含义与加载训练集类似，不过train=False表示加载的是测试集数据
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

# 定义类别标签，CIFAR10数据集包含10个类别，这里对应每个类别的名称
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# VGG16模型定义部分
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        # 特征提取部分，使用nn.Sequential将多个层按顺序组合在一起，方便构建网络结构
        self.features = nn.Sequential(
            # conv1两次[3,3]卷积网络，输出的特征层为64，输出为(32,32,64)
            # 第一个卷积层，输入通道数为3（对应CIFAR10图像的RGB三个通道），输出通道数为64，卷积核大小为3x3，填充为1
            # 填充为1可以保证卷积后图像尺寸不变（在输入尺寸为32x32时）
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            # 使用ReLU激活函数进行非线性变换，inplace=True表示直接在原张量上进行操作，节省内存空间
            nn.ReLU(inplace=True),
            # 第二个卷积层，输入通道数为64（上一层的输出通道数），输出通道数为64，卷积核大小和填充方式同前
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 最大池化层，池化核大小为2x2，步长为2，会使图像尺寸减半（从32x32变为16x16）
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv2两次[3,3]卷积网络，输出的特征层为128，输出为(16,16,128)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv3三次[3,3]卷积网络，输出的特征层为256，输出为(8,8,256)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv4三次[3,3]卷积网络，输出的特征层为512，输出为(4,4,512)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # conv5三次[3,3]卷积网络，输出的特征层为512，输出为(2,2,512)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 自适应平均池化层，会将输入的任意大小的特征图转换为固定大小(7, 7)的特征图
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # 分类器部分，同样使用nn.Sequential组合多个全连接层及相关操作
        self.classifier = nn.Sequential(
            # 第一个全连接层，将展平后的特征向量（长度为512 * 7 * 7）映射到4096维空间
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            # Dropout层，以一定概率（默认0.5）随机将神经元的输出置为0，防止过拟合，在训练时起作用
            nn.Dropout(),
            # 第二个全连接层，将4096维空间再映射到4096维空间
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # 最后一个全连接层，输出维度为num_classes（CIFAR10中为10），对应10个类别
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # 前向传播函数，定义数据在网络中的流动顺序
        x = self.features(x)
        x = self.avgpool(x)
        # 将经过平均池化后的特征图展平为一维向量，方便输入到全连接层中
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 训练函数定义
def train(model, device, trainloader, optimizer, criterion, epoch):
    """
    训练函数，用于对模型进行一个epoch的训练

    参数:
    model: 要训练的神经网络模型
    device: 训练使用的计算设备（GPU或CPU）
    trainloader: 训练集数据加载器，按批次提供训练数据
    optimizer: 优化器，用于更新模型的参数
    criterion: 损失函数，用于计算模型输出与真实标签之间的差异
    epoch: 当前训练的轮次编号
    """
    # 将模型设置为训练模式，这会影响一些层（如Dropout、BatchNorm等）的行为，使其在训练时按规则工作
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        # 将数据和目标标签移动到指定的计算设备（GPU或CPU）上，确保在相应设备上进行计算
        data, target = data.to(device), target.to(device)
        # 在每次迭代开始前，清空之前累积的梯度信息，避免梯度累加影响本次参数更新
        optimizer.zero_grad()
        # 进行前向传播，将输入数据传入模型得到输出
        output = model(data)
        # 计算模型输出与真实标签之间的损失，使用之前定义的损失函数
        loss = criterion(output, target)
        # 进行反向传播，根据计算出的损失，计算每个参数的梯度，用于后续的参数更新
        loss.backward()
        # 根据计算得到的梯度，使用优化器更新模型的参数
        optimizer.step()

        running_loss += loss.item()
        # 获取预测结果中概率最大的类别索引，作为模型的预测类别
        _, predicted = output.max(1)
        # 统计本批次中总的样本数量
        total += target.size(0)
        # 统计本批次中预测正确的样本数量，通过比较预测类别和真实标签是否一致来判断
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), running_loss / 100))
            running_loss = 0.0

    print('Train Epoch: {} \tAccuracy: {:.4f}'.format(epoch, correct / total))

# 验证函数定义
def validate(model, device, testloader, criterion):
    """
    验证函数，用于评估模型在测试集上的性能

    参数:
    model: 要评估的神经网络模型
    device: 评估使用的计算设备（GPU或CPU）
    testloader: 测试集数据加载器，按批次提供测试数据
    criterion: 损失函数，用于计算模型输出与真实标签之间的差异
    """
    # 将模型设置为评估模式，使一些层（如Dropout等）按评估时的规则工作，例如Dropout不再随机丢弃神经元
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        running_loss / len(testloader), correct / total))
    return correct / total

# 主函数与训练、推理流程
def main():
    # 根据是否有可用的GPU来选择计算设备，如果有GPU则使用'cuda'，否则使用'cpu'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化VGG16模型，并将其移动到选择的计算设备上
    model = VGG16().to(device)
    # 定义损失函数，这里使用交叉熵损失函数，适用于多分类任务，它会自动将模型输出转换为概率分布并计算与真实标签的差异
    criterion = nn.CrossEntropyLoss()
    # 定义优化器，使用随机梯度下降（SGD）算法，传入模型的参数、学习率、动量以及权重衰减系数等参数
    # 学习率决定每次参数更新的步长大小，动量可以帮助加速收敛并减少在局部最小值附近的震荡，权重衰减用于防止过拟合
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    epochs = 10
    best_acc = 0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        # 调用训练函数对模型进行一个epoch的训练，传入模型、设备、训练集数据加载器、优化器、损失函数以及当前轮次编号
        train(model, device, trainloader, optimizer, criterion, epoch)
        # 调用验证函数评估模型在测试集上的性能，传入模型、设备、测试集数据加载器以及损失函数
        acc = validate(model, device, testloader, criterion)
        end_time = time.time()
        print('Epoch {} took {:.2f} seconds'.format(epoch, end_time - start_time))

        if acc > best_acc:
            best_acc = acc
            # 如果当前验证准确率高于之前保存的最佳准确率，则保存当前模型的参数到文件中
            torch.save(model.state_dict(), 'vgg16_cifar10_best.pth')

    print('Best accuracy: {:.4f}'.format(best_acc))

    # 推理示例（加载最佳模型进行预测）
    model.load_state_dict(torch.load('vgg16_cifar10_best.pth'))
    model.eval()
    with torch.no_grad():
        dataiter = iter(testloader)
        images, labels = next(dataiter)
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(len(images))))
        print('Ground truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(images))))


if __name__ == '__main__':
    main()
