# -*- coding: utf-8 -*-
# time: 2024/11/14 10:58
# file: tf.py
# author: flame
""" 导入numpy库，用于生成随机数据和数学运算。 """
import numpy as np

""" 导入tensorflow库，用于构建和训练神经网络模型。 """
import tensorflow as tf

""" 导入matplotlib.pyplot库，用于绘制图形。 """
import matplotlib.pyplot as plt

"""
生成200个随机点，定义一个两层神经网络模型，使用梯度下降优化器训练模型，
并在训练完成后绘制原始数据点和预测结果的图形。
"""
""" 使用numpy的linspace函数生成从-0.5到0.5之间的200个等间距点，形状为(200, 1)。 """
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]

""" 使用numpy的random.normal函数生成均值为0.0，标准差为0.02的正态分布噪声，形状与x_data相同。 """
noise = np.random.normal(0.0, 0.02, x_data.shape)

""" 生成目标数据y_data，通过将x_data平方并加上噪声得到，形状为(200, 1)。 """
y_data = np.square(x_data) + noise

""" 定义一个占位符x，用于接收输入数据，数据类型为float32，形状为(None, 1)，表示可以接受任意数量的样本，每个样本有一个特征。 """
x = tf.placeholder(tf.float32, [None, 1])

""" 定义一个占位符y，用于接收目标数据，数据类型为float32，形状为(None, 1)，表示可以接受任意数量的样本，每个样本有一个标签。 """
y = tf.placeholder(tf.float32, [None, 1])

""" 定义第一层神经网络的权重，使用随机正态分布初始化，形状为(1, 10)，表示输入层有一个节点，隐藏层有10个节点。 """
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))

""" 定义第一层神经网络的偏置，初始值为0，形状为(1, 10)，表示隐藏层有10个节点。 """
biases_L1 = tf.Variable(tf.zeros([1, 10]))

""" 计算第一层神经网络的线性组合，即Wx + b，其中W是权重矩阵，x是输入数据，b是偏置向量。 """
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1

""" 对第一层神经网络的线性组合应用tanh激活函数，得到隐藏层的输出。 """
prediction = tf.nn.tanh(Wx_plus_b_L1)

""" 定义第二层神经网络的权重，使用随机正态分布初始化，形状为(10, 1)，表示隐藏层有10个节点，输出层有一个节点。 """
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))

""" 定义第二层神经网络的偏置，初始值为0，形状为(1, 1)，表示输出层有一个节点。 """
biases_L2 = tf.Variable(tf.zeros([1, 1]))

""" 计算第二层神经网络的线性组合，即Wx + b，其中W是权重矩阵，x是隐藏层的输出，b是偏置向量。 """
Wx_plus_b_L2 = tf.matmul(prediction, Weights_L2) + biases_L2

""" 对第二层神经网络的线性组合应用tanh激活函数，得到最终的预测输出。 """
prediction = tf.nn.tanh(Wx_plus_b_L2)

""" 定义损失函数，计算预测值与真实值之间的均方误差。 """
loss = tf.reduce_mean(tf.square(y - prediction))

""" 定义优化器，使用梯度下降法，学习率为0.1，最小化损失函数。 """
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

""" 创建一个TensorFlow会话，用于运行计算图。 """
with tf.Session() as sess:
    """ 初始化所有变量，包括权重和偏置。 """
    sess.run(tf.global_variables_initializer())

    """ 训练2000次，每次迭代更新模型参数。 """
    for i in range(2000):
        """ 运行训练步骤，传入训练数据x_data和y_data。 """
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    """ 获取训练后的预测值，传入x_data作为输入。 """
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    """ 创建一个新的图形窗口。 """
    plt.figure()

    """ 绘制散点图，显示原始数据点。 """
    plt.scatter(x_data, y_data)

    """ 绘制预测曲线，颜色为红色，线宽为5。 """
    plt.plot(x_data, prediction_value, 'r-', lw=5)

    """ 显示图形。 """
    plt.show()
