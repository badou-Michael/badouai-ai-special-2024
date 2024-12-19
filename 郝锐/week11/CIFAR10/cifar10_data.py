#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/12/2 16:33
# @Author: Gift
# @File  : tf_train.py 
# @IDE   : PyCharm
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
tf.disable_eager_execution()
# 使用numpy生成200个随机点,均匀分布在-0.5到0.5之间
x_data = np.linspace(-0.5, 0.5, 200) #这是一个包含200个数据的数组
print(x_data)
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis] # 将一维数组转换为二维数组，200行1列的数组
print(x_data)
noise = np.random.normal(0, 0.02, x_data.shape) #均值为0，标准差为0.02的噪声，形状和x_data的一样
y_data = np.square(x_data) + noise #每个元素的平方值，再加上噪声

# 定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))  # 加入偏置项
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)  # 加入激活函数

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))  # 加入偏置项
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)  # 加入激活函数

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y - prediction))
# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 训练2000次
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
    plt.show()
