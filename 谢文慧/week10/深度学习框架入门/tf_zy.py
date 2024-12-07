import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#使用numpy生成200个随机点
# [:,np.newaxis],将数列变成矩阵，n行1列的矩阵
xdata = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.1,xdata.shape)
ydata = np.square(xdata) + noise

#定义两个placeholder存放输入数据
#  tf.placeholder(dtype, shape=None, name=None):此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
# dtype：数据类型。常用的是tf.float32, tf.float64等数值类型
# shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2, 3], [None, 3] 表示列是3，行不定
# name：名称。
# x和y 定义类型是float32，列是1，行不确定
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#定义神经网络中间层
# tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# shape: 输出张量的形状，必选
# mean: 正态分布的均值，默认为0
# stddev: 正态分布的标准差，默认为1.0
# dtype: 输出的类型，默认为tf.float32
# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
# name: 操作的名称
# 输入层和隐藏层的权重
wih = tf.Variable(tf.random_normal([1,10]))
# 加入偏置项
biases = tf.Variable(tf.zeros([1,10]))
# 加权结果，隐藏层的输入信号
Wihx_plus_b = tf.matmul(x,wih)+biases
# 隐藏层输出信号,激活函数
hidden_out = tf.nn.tanh(Wihx_plus_b)

#定义神经网络输出层
who = tf.Variable(tf.random_normal([10,1]))
biases2 = tf.Variable(tf.zeros([1,1]))
whox_plus_b = tf.matmul(hidden_out,who) + biases2
out = tf.nn.tanh(whox_plus_b)

#定义损失函数（均方差函数）
loss= tf.reduce_mean(tf.square(y-out))
#定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
#     变量初始化
    sess.run(tf.global_variables_initializer())
# 训练2000次
    for i in range(2000):
        sess.run(train_step,feed_dict={x:xdata,y:ydata})
        print(x)
        print(y)
        print('----------')
#     预测推理
    prediction = sess.run(out,feed_dict={x :xdata})


    # 画图
    plt.figure()
    plt.scatter(xdata, ydata)  # 散点是真实值
    plt.plot(xdata, prediction, 'r-', lw=5)  # 曲线是预测值
    plt.show()
