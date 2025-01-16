import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 定义原始 AlexNet 模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
            # 最后一层不需要 Softmax，因为 CrossEntropyLoss 已经包含了
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.img_labels = []
        with open(txt_file, "r") as f:
            lines = f.readlines()
        # 打乱行，增加数据随机性
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        for line in lines:
            name, label = line.strip().split(';')
            self.img_labels.append((name, int(label)))
        self.transform = transform
        self.img_dir = r".\Course_CV\Week11\alexnet\alexnet_train_data\train"

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name, label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, label

# 图像预处理和数据增强
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
])

if __name__ == "__main__":
    # 加载数据集
    dataset = CustomDataset(
        txt_file=r".\Course_CV\Week11\alexnet\AlexNet-Keras-master\data\dataset.txt",
        transform=transform
    )

    # 划分训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 定义数据加载器
    batch_size = 1536
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型并移动到设备
    model = AlexNet(num_classes=2).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 使用 Adam 优化器

    # 定义学习率调整策略
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 日志和模型保存路径
    log_dir = r".\Course_CV\Week11\alexnet"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 训练模型
    num_epochs = 50
    best_acc = 0.0
    epochs_no_improve = 0  # 记录验证损失未降低的次数
    n_epochs_stop = 10     # 早停的阈值

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # 训练循环
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 清零梯度，反向传播，更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / train_size
        epoch_acc = running_corrects.double() / train_size

        print('Epoch {}/{} Train Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, epoch_acc))

        # 验证模型
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_loss = val_running_loss / val_size
        val_acc = val_running_corrects.double() / val_size

        print('Epoch {}/{} Val Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs, val_loss, val_acc))

        # 学习率调整
        scheduler.step(val_acc)

        # 检查验证准确率是否提升
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0  # 重置早停计数器
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model_torch.pth'))
        else:
            epochs_no_improve += 1

        # 检查是否满足早停条件
        if epochs_no_improve >= n_epochs_stop:
            print('早停触发，停止训练')
            break

    print('训练完成，最佳验证准确率：{:.4f}'.format(best_acc))