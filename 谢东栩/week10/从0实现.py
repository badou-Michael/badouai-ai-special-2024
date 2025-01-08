import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import random


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate

        # 初始化权重
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # 前向传播
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))

        # 更新权重
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                     hidden_outputs.T)
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                     inputs.T)

    def query(self, inputs):
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


# 初始化网络
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读取训练数据
with open("dataset/mnist_train.csv", 'r') as f:
    training_data_list = f.readlines()

# 训练模型
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

# 读取测试数据
with open("dataset/mnist_test.csv", 'r') as f:
    test_data_list = f.readlines()

# 测试模型并计算准确率
scorecard = []
for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    predicted_label = np.argmax(outputs)
    if predicted_label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)

# 计算准确率
scorecard_array = np.asarray(scorecard)
accuracy = scorecard_array.sum() / scorecard_array.size
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 随机选取一张图片展示并预测
random_record = random.choice(test_data_list)
all_values = random_record.split(',')
correct_label = int(all_values[0])

# 显示随机图片
image = np.asfarray(all_values[1:]).reshape(28, 28)
plt.imshow(image, cmap='Greys', interpolation='None')
plt.title(f"Correct Label: {correct_label}")
plt.show()

# 对随机图片进行预测
inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
outputs = n.query(inputs)
predicted_label = np.argmax(outputs)

print(f"Predicted Label: {predicted_label}")
