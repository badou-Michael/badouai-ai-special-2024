import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

'''
tf.random_normal是一个TensorFlow函数，它从一个正态分布（也称为高斯分布）中随机抽取值来填充一个张量。这个函数的参数是你想要的张量的形状，以及可选的均值和标准差。
创建一个形状为[1, 10]的张量：这个张量有1行10列，可以想象成一个包含10个元素的数组，每个元素都是一个权重值。
用正态分布的随机数填充这个张量：tf.random_normal默认的均值（mean）是0，标准差（stddev）是1。这意味着每个权重值都是从一个均值为0、标准差为1的正态分布中随机抽取的。
'''
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
    '''
    for i in range(2000):循环2000次，每次迭代都通过sess.run(train_step, feed_dict={x: x_data, y: y_data})更新模型参数，以最小化损失函数。
    plt.plot(x_data, prediction_value, 'r-', lw=5)绘制预测值的曲线图，红色实线，线宽为5。
    '''
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值 lw是linewidth的缩写，用于指定线的宽度。
    plt.show()
# plt.scatter() 用于绘制散点图，这种图表类型非常适合展示两个变量之间的关系。plt.plot() 用于绘制线图，非常适合展示数据随某个连续变量（如时间）的变化趋势。