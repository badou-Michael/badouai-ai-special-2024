#-*- coding:utf-8 -*-
# author: 王博然
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成200个从-0.5 到 0.5 均匀分布一维数组
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  # (200,1)
noise = np.random.normal(0, 0.02, x_data.shape) # 正态分布, 标准差 0.02
y_data = np.square(x_data) + noise

# 定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32, [None,1])
y = tf.placeholder(tf.float32, [None,1])

# 定义神经网络中间层
weight_l1 = tf.Variable(tf.random.normal([1,10]))  # 生成一个形状为 [1, 10] 的张量
biases_l1 = tf.Variable(tf.zeros([1,10]))
wx_plusb_l1 = tf.matmul(x, weight_l1) + biases_l1
l1 = tf.nn.tanh(wx_plusb_l1)

# 定义神经网络输出层
weight_l2 = tf.Variable(tf.random.normal([10,1]))  # 生成一个形状为 [10, 1] 的张量
biases_l2 = tf.Variable(tf.zeros([1,1]))
wx_plusb_l2 = tf.matmul(l1, weight_l2) + biases_l2
prediction = tf.nn.tanh(wx_plusb_l2)

# 定义损失函数 (均方差)
loss = tf.reduce_mean(tf.square(y - prediction))
# 定义反向传播算法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000): # 训练2000次
        sess.run(train_step, feed_dict={x:x_data, y:y_data})

    predict_y = sess.run(prediction, feed_dict={x:x_data})
    print(prediction)
    print(predict_y)

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data) # 散点是真实值
    plt.plot(x_data, predict_y, 'r-', lw=5) # 曲线是预测值
    plt.show()