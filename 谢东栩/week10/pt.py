import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random

# 数据加载
def load_data(file_path):
    with open(file_path, 'r') as f:
        data_list = f.readlines()
    return data_list

train_data_list = load_data("dataset/mnist_train.csv")
test_data_list = load_data("dataset/mnist_test.csv")

# 数据预处理
def preprocess_data(data_list):
    inputs = []
    labels = []
    for record in data_list:
        all_values = record.split(',')
        inputs.append((np.asfarray(all_values[1:]) / 255.0).tolist())
        labels.append(int(all_values[0]))
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

train_inputs, train_labels = preprocess_data(train_data_list)
test_inputs, test_labels = preprocess_data(test_data_list)

# 数据标准化
mean = torch.mean(train_inputs, dim=0)
std = torch.std(train_inputs, dim=0)
train_inputs = (train_inputs - mean) / std
test_inputs = (test_inputs - mean) / std

# 构建 PyTorch 模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(784, 512)
        self.hidden2 = nn.Linear(512, 200)
        self.output = nn.Linear(200, 10)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)  # Dropout 层

    def forward(self, x):
        x = self.activation(self.hidden1(x))
        x = self.dropout(x)
        x = self.activation(self.hidden2(x))
        x = self.dropout(x)
        x = self.softmax(self.output(x))
        return x

model = NeuralNetwork()

# 定义损失函数和优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 训练模型
epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_inputs)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# 测试模型
model.eval()
with torch.no_grad():
    test_outputs = model(test_inputs)
    predicted_labels = torch.argmax(test_outputs, dim=1)
    accuracy = (predicted_labels == test_labels).sum().item() / len(test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 随机选择一张图片展示并预测
random_index = random.randint(0, len(test_inputs) - 1)
random_image = test_inputs[random_index].reshape(28, 28)
plt.imshow(random_image, cmap='Greys', interpolation='None')
plt.title(f"Correct Label: {test_labels[random_index].item()}")
plt.show()

# 对随机图片进行预测
random_input = test_inputs[random_index].unsqueeze(0)
predicted_label = torch.argmax(model(random_input)).item()
print(f"Predicted Label: {predicted_label}")
