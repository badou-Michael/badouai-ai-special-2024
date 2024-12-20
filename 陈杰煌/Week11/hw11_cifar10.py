# 该文件使用 PyTorch 对 CIFAR-10 进行重构，
# 对 CIFAR-10 数据集进行训练和测试，其中训练数据集进行数据增强处理，测试数据集不进行数据增强处理

import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from torch.amp import autocast, GradScaler

# 设置用于训练和评估的样本总数
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000

# 定义一个用于返回读取的 CIFAR-10 数据的数据集类
class CIFAR10Dataset(data.Dataset):
    def __init__(self, data_dir, batch_size, distorted):
        self.batch_size = batch_size
        if distorted:
            # 拼接训练数据的文件地址
            self.filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
        else:
            # 拼接测试数据的文件地址
            self.filenames = [os.path.join(data_dir, "test_batch.bin")]
        self.data = []
        self.labels = []
        for filename in self.filenames:
            with open(filename, 'rb') as f:
                dict = self._unpickle(f)
                self.data.append(dict['data'])
                self.labels += dict['labels']
        self.data = np.concatenate(self.data)
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose(0, 2, 3, 1)  # 转换为 (N, H, W, C)

        if distorted:
            # 定义数据增强的变换
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(24),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.8, contrast=0.8),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            # 定义测试数据的变换
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(24),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def _unpickle(self, file):
        import numpy as np
        import pickle
        try:
            # CIFAR-10 数据集的二进制文件格式
            dict = {}
            dict['data'] = np.frombuffer(file.read(10000 * 3072), dtype=np.uint8)
            dict['data'] = dict['data'].reshape(10000, 3, 32, 32)
            dict['data'] = dict['data'].transpose(0, 2, 3, 1)  # 转换为 (N, H, W, C)
            dict['labels'] = np.frombuffer(file.read(10000), dtype=np.uint8).tolist()
            # 确保标签值在 0 到 9 之间
            dict['labels'] = [label if 0 <= label < 10 else 0 for label in dict['labels']]
            return dict
        except Exception as e:
            print(f"Error unpickling file: {file}")
            print(e)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        img = self.transform(img)
        return img, label

# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)
        self.pool = nn.MaxPool2d(3, 2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 6 * 6, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # 动态获取批量大小并展平特征图
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    # 定义损失函数和优化器，使用L2正则化
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.004)

    # 训练网络
    scaler = GradScaler('cuda')
    max_steps = 100  # 假设最大训练步数为100
    batch_size = 20000  # 增大批量大小
    data_dir = "./Course_CV/Week11/cifar/cifar_data"
    trainloader = torch.utils.data.DataLoader(CIFAR10Dataset(data_dir, batch_size, True), batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(CIFAR10Dataset(data_dir, batch_size, False), batch_size=batch_size, shuffle=False, num_workers=2)

    for step in range(max_steps):
        net.train()
        start_time = time.time()
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # 前向传播和计算损失
            with autocast(device_type='cuda'):
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        duration = time.time() - start_time

        # 每个 step 打印一次训练信息
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        print("step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)" % (step, loss.item(), examples_per_sec, sec_per_batch))

    print("训练完成")

    # 计算最终的正确率
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 打印正确率信息
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')