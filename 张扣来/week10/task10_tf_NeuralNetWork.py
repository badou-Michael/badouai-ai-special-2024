import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

# 使用np生成200个随机点，均匀分布在-0.5~0.5之间，np.newaxis用于转换数据为二维数组
x_data = np.linspace(-0.5,0.5,200)[:, np.newaxis]
# 生成一个正态分布的随机数数组，其均值为0，标准差为0.02，形状与 x_data 相同。
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

# 定义占位符，存储数据
# [None, 1]：这指定了占位符的形状。None表示该维度的大小可以是任何值，
# 这使得占位符可以接收任何数量的数据点。1 表示每个数据点是一个一维的向量。
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
'''
创建一个权重变量 Weights_L1，用于第一层（输入层到隐藏层）。
tf.random_normal 函数生成一个均值为0，标准差为1的正态分布随机数矩阵，形状为 [1, 10]，
标准的正态分布一般均值都为0，标准差为1
即输入层有1个节点，隐藏层有10个节点。
'''
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
# 数据与权重矩阵相乘，加上偏置量
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) +biases_L1
# 激活函数得到第一层输出值，tf.nn.tanh双曲正切激活函数
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络的输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
# 隐藏层的10个节点和权重相乘加上偏置量
Wx_plus_b_L2 =  tf.matmul(L1, Weights_L2) + biases_L2
# 输出
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 损失函数的定义
loss = tf.reduce_mean(tf.square(y - prediction))
# 定义反向传播算法（使用梯度下降算法训练）,最小化损失函数 loss
'''
梯度下降优化器（Gradient Descent Optimizer），用于在训练过程中更新模型的参数。
0.1 是学习率（learning rate），它控制着参数更新的步长。
.minimize(loss)：这个方法告诉优化器，我们希望最小化 loss 这个损失函数。
'''
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 初始化所有全局变量
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step, feed_dict={x:x_data,y:y_data})
    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x:x_data})

    plt.figure()
    plt.scatter(x_data, y_data)
    # 图表上绘制一条红色的实线，宽度为5
    plt.plot(x_data,prediction_value,'r-',lw = 5)
    plt.show()
