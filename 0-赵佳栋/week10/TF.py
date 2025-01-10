#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：TF.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/12/06 15:32
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 使用 numpy 生成 200 个在 -0.5 到 0.5 之间的等间距数据点，并将其转换为二维数组
# 这样做是为了将数据点作为输入特征，方便后续的矩阵运算
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]


# 生成噪声数据，这些噪声服从均值为 0，标准差为 0.02 的正态分布
# 噪声的形状与 x_data 相同，目的是为了给目标数据添加一些随机性
noise = np.random.normal(0, 0.02, x_data.shape)


# 生成目标数据，这里是将 x_data 的每个元素平方后加上噪声
# 这样生成的数据具有一定的非线性关系，符合一个二次函数的形式
y_data = np.square(x_data) + noise


# 定义两个占位符，用于在训练过程中接收输入数据和目标数据
# x 是输入数据的占位符，形状为 [None, 1]，表示可以接收任意数量的样本，每个样本有 1 个特征
x = tf.placeholder(tf.float32, [None, 1])


# y 是目标数据的占位符，形状为 [None, 1]，表示可以接收任意数量的样本，每个样本有 1 个目标值
y = tf.placeholder(tf.float32, [None, 1])


# 定义神经网络的中间层


# 中间层的权重矩阵，形状为 [1, 10]，使用随机正态分布进行初始化
# 这表示从输入层（1 个神经元）到中间层（10 个神经元）的连接权重
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))


# 中间层的偏置项，形状为 [1, 10]，初始化为 0
# 偏置项可以让模型更好地拟合数据，提供一定的偏移调整能力
biases_L1 = tf.Variable(tf.zeros([1, 10]))


# 计算中间层的线性组合，即输入数据与中间层权重矩阵相乘，并加上中间层的偏置项
# 这里使用了矩阵乘法和加法运算，符合线性代数的计算规则
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1


# 使用双曲正切函数（tanh）作为中间层的激活函数
# 激活函数可以引入非线性，使神经网络能够拟合非线性关系
L1 = tf.nn.tanh(Wx_plus_b_L1)


# 定义神经网络的输出层


# 输出层的权重矩阵，形状为 [10, 1]，使用随机正态分布进行初始化
# 这表示从中间层（10 个神经元）到输出层（1 个神经元）的连接权重
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))


# 输出层的偏置项，形状为 [1, 1]，初始化为 0
biases_L2 = tf.Variable(tf.zeros([1, 1]))


# 计算输出层的线性组合，即中间层的输出与输出层权重矩阵相乘，并加上输出层的偏置项
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2


# 使用双曲正切函数（tanh）作为输出层的激活函数
# 输出层的激活函数将输出值映射到一个范围，这里使用 tanh 函数是为了使输出在 -1 到 1 之间
prediction = tf.nn.tanh(Wx_plus_b_L2)


# 定义损失函数，使用均方误差（MSE）来衡量预测值和真实值之间的差异


# 计算预测值和真实值之间的平方差，并取平均值
# 这是一种常用的损失函数，用于衡量回归问题的误差大小
loss = tf.reduce_mean(tf.square(y - prediction))


# 定义反向传播算法，使用梯度下降优化器来最小化损失函数


# 创建一个梯度下降优化器，学习率为 0.1
# 优化器会根据计算得到的梯度来更新网络中的权重和偏置项，以最小化损失函数
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# 创建一个 TensorFlow 会话，用于执行计算图
with tf.Session() as sess:


    # 初始化所有变量
    # 这是 TensorFlow 的必要步骤，确保变量在使用前被正确初始化
    sess.run(tf.global_variables_initializer())


    # 进行 2000 次训练迭代
    for i in range(2000):


        # 运行优化器，将输入数据和目标数据传递给占位符
        # 通过 feed_dict 将实际的数据喂入计算图中，以便计算损失和更新参数
        sess.run(train_step, feed_dict={x: x_data, y: y_data})


    # 获取预测值，将输入数据传递给预测操作
    # 这里的预测值是经过训练后的神经网络对输入数据的输出
    prediction_value = sess.run(prediction, feed_dict={x: x_data})


    # 绘制图像


    # 创建一个新的图像窗口
    plt.figure()


    # 绘制散点图，显示真实数据点
    # 散点图可以直观地展示输入数据和目标数据的分布情况
    plt.scatter(x_data, y_data)


    # 绘制曲线，显示预测结果
    # 用红色实线绘制，线宽为 5，展示训练后的神经网络对输入数据的预测输出
    plt.plot(x_data, prediction_value, 'r-', lw=5)


    # 显示图像
    plt.show()