#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/12/2 14:35
# @Author: Gift
# @File  : NeuralNetWork.py 
# @IDE   : PyCharm
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
# 加载mnist 数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#将数据展开为一维向量并且归一化
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255 #数据类型转换为float32
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255 #数据类型转换为float32
#将标签进行one-hot编码 np.eye 生成一个10*10的单位矩阵，比如labels为1，则返回[0,1,0,0,0,0,0,0,0,0]
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]
print(train_labels[0])
print(train_labels.shape[0])
train_labels = one_hot_encode(train_labels, 10)
print(train_labels[0])
test_labels = one_hot_encode(test_labels, 10)
#定义神经网络的类
class NerualNetwork():
    def __init__(self, input_size, hidden_size, output_size): #初始化神经网络
        #设置权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01 #输入层到隐藏层的权重
        self.b1 = np.zeros((1, hidden_size)) #输入层到隐藏层的偏置
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01 #隐藏层到输出层的权重
        self.b2 = np.zeros((1, output_size)) #隐藏层到输出层的偏置
    def relu(self, x): #激活函数
        return np.maximum(0, x)
    #定义softmax
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    #定义前向传播
    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1 #输入层到隐藏层的线性变换
        self.a1 = self.relu(self.z1) #输入层到隐藏层的激活函数
        self.z2 = np.dot(self.a1, self.W2) + self.b2 #隐藏层到输出层的线性变换
        self.a2 = self.softmax(self.z2) #隐藏层到输出层的激活函数
        return self.a2
    #损失函数，分类用交叉熵
    def cross_entropy_loss(self, y_true, y_pred):
        total = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / total
        return loss
    #定义反向传播
    def backward(self, x, y_true, learning_rate): #输入数据，真实标签，学习率
        m = x.shape[0] #样本数量
        #计算输出层的梯度
        dZ2 = self.a2 - y_true #输出层的梯度
        dW2 = np.dot(self.a1.T, dZ2) / m #输出层到隐藏层的权重梯度
        dB2 = np.sum(dZ2, axis=0, keepdims=True) / m #输出层到隐藏层的偏置梯度
        #计算隐藏层的梯度
        dZ1 = np.dot(dZ2, self.W2.T) * (self.a1 > 0) #输入层到隐藏层的梯度
        dW1 = np.dot(x.T, dZ1) / m #输入层到隐藏层的权重梯度
        dB1 = np.sum(dZ1, axis=0, keepdims=True) / m #输入层到隐藏层的偏置梯度
        return dW1, dB1, dW2, dB2
    #更新权重和偏置
    def update_weight(self, dW1, dB1, dW2, dB2, learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * dB1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * dB2
#设置神经网络的超参
input_size = 28 * 28
hidden_size = 128
output_size = 10
learning_rate = 0.01
epochs = 20
batch_size = 128
# 创建神经网络实例
neural_network = NerualNetwork(input_size, hidden_size, output_size)
for epoch in range(epochs):
    epoch_loss = 0
    num_batches = len(train_images) // batch_size
    for i in range(num_batches):
        #获取批次数据
        batch_images = train_images[i * batch_size : (i + 1) * batch_size]
        batch_labels = train_labels[i * batch_size : (i + 1) * batch_size]
        #前向传播
        y_pred = neural_network.forward(batch_images)
        #计算损失
        loss = neural_network.cross_entropy_loss(batch_labels, y_pred)
        epoch_loss += loss
        #反向传播
        dW1, dB1, dW2, dB2 = neural_network.backward(batch_images, batch_labels, learning_rate)
        #更新权重和偏置
        neural_network.update_weight(dW1, dB1, dW2, dB2, learning_rate)
    #打印每个epoch的损失
    epoch_loss /= num_batches
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/num_batches}")
#在评测集上评估模型
y_pred_test = neural_network.forward(test_images)
predict_labels = np.argmax(y_pred_test, axis=1)
true_labels = np.argmax(test_labels, axis=1)
accuracy = np.sum(predict_labels == true_labels) / len(true_labels)
print(f"测试集Accuracy: {accuracy}")
test_pred = neural_network.forward(test_images[0])
test_pred_labels = np.argmax(test_pred, axis=1)
print(f"预测标签索引: {test_pred_labels}")
print(f"真实标签索引: {test_labels[0]}")
