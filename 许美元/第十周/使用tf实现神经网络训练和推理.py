

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点，并将 np.linspace 生成的一维数组转换为二维数组，其中每个数字都是一个行向量。
x_data = np.linspace(start=-0.5, stop=0.5, num=200)[:, np.newaxis]
## 生成正态分布（高斯分布）的随机样本。
# 0 是正态分布的均值（mean）。
# 0.02 是正态分布的标准差（standard deviation）。
# x_data.shape 指定了生成噪声数组的形状，与 x_data 相同，确保每个 x 值都有一个对应的噪声值。
noise = np.random.normal(0, 0.02, x_data.shape)
# np.square 计算 x_data 中每个元素的平方，生成二次函数的值。
# +noise 将第二步生成的噪声添加到二次函数的值上，创建了一个带有噪声的二次函数数据集 y_data。
y_data = np.square(x_data) + noise

## 定义两个placeholder存放输入数据
# 它是一个一维的、数据类型为 32 位浮点数的数组，其第一个维度（通常是批量大小）可以是任意的。
# 这意味着 x 可以接收任何数量的样本作为输入，每个样本是一个一维数组。
x = tf.compat.v1.placeholder(tf.float32, [None, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 1])

## 定义神经网络中间层
# 生成一个 1x10 的矩阵，即 10 个权重值，对应于 10 个输出节点（或特征）。
# tf.random_normal 函数用于生成一个正态分布的随机数
weights_l1 = tf.Variable(tf.random.normal([1, 10]))
biases_l1 = tf.Variable(tf.zeros([1, 10]))
# tf.matmul 执行矩阵乘法
wx_plus_b_l1 = tf.matmul(x, weights_l1) + biases_l1
l1 = tf.nn.tanh(wx_plus_b_l1)  # 激活函数

## 定义神经网络输出层
weights_l2 = tf.Variable(tf.random_normal([10, 1]))
biases_l2 = tf.Variable(tf.zeros([1, 1]))
wx_plus_b_l2 = tf.matmul(l1, weights_l2) + biases_l2
prediction = tf.nn.tanh(wx_plus_b_l2) # 激活函数

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y - prediction))
# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.compat.v1.Session() as sess:
    # 变量初始化
    sess.run(tf.compat.v1.global_variables_initializer())
    # 训练2000次
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
