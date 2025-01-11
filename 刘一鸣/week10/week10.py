'''
第十周作业
1.从零实现训练过程；
2.使用tf实现神经网络训练和推理；
3.使用pytorch实现手写数字识别

data数据是上课给的数据
'''

#1、使用tensorflow
#版本2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理：归一化并调整形状
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255

# 将标签转换为独热编码
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 构建神经网络模型
model = models.Sequential()

# 添加卷积层和池化层
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 展平并连接全连接层
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
print("Training the model...")
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

# 评估模型
print("Evaluating the model...")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# 推理：使用模型进行预测
sample = x_test[:5]  # 取测试集中的前 5 个样本
predictions = model.predict(sample)

# 输出预测结果
for i, pred in enumerate(predictions):
    print(f"Sample {i + 1}: Predicted Label: {tf.argmax(pred).numpy()}, True Label: {tf.argmax(y_test[i]).numpy()}")

#2、使用pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#1、加载MNIST数据集
def load_data(batch_size=64):
    transform = transforms.Compose(
        [transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
         transforms.Normalize([0, ], [1, ])])
    #加载训练集
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #加载测试集
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

#2、定义神经网络
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # 输入层到隐藏层1
        self.fc1 = nn.Linear(28 * 28, 512)
        # 隐藏层1到隐藏层2
        self.fc2 = nn.Linear(512, 256)
        # 隐藏层2到输出层
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平图像为一维张量
        x = torch.relu(self.fc1(x))  # ReLU 激活函数
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # 输出层（未加 Softmax，CrossEntropyLoss 内部会处理）
        return x

#3、训练模型
def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新权重
            optimizer.step()
            # 累积损失
            running_loss += loss.item()
            if (i + 1) % 100 == 0:  # 每 100 个批次打印一次损失
                print(
                    f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

#4、评估模型
def evaluate_model(model, test_loader):
    model.eval()  # 切换到评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算以提高效率
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # 获取预测的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total:.2f}%")

#5、主函数
if __name__ == "__main__":
    # 加载数据
    train_loader, test_loader = load_data(batch_size=64)
    # 初始化模型
    model = MNISTModel()
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练模型
    train(model, train_loader, criterion, optimizer, epochs=5)
    # 评估模型
    evaluate_model(model, test_loader)
