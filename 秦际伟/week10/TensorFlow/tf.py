import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow 版本", tf.__version__)

# 使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]     # 生成 -0.5到0.5的200个数据，并增加一个维度，方便后面矩阵运算
noise = np.random.normal(0, 0.02, x_data.shape)     # 生成与x_data形状相同的正态分布的噪声数据，均值为0，方差为0.02
y_data = np.square(x_data) + noise                  # 生成目标函数，  y = x^2 + 噪声数据

# 定义两个 placeholder 存放输入数据
x = tf.placeholder(tf.float32, [None, 1])  # 输入数据，None表示任意行，1表示列，因为只有一个特征
y = tf.placeholder(tf.float32, [None, 1])   # 输出数据，None表示任意行，1表示列，因为只有一个特征

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))   # 定义第一层权重，1行10列，随机生成
biases_L1 = tf.Variable(tf.zeros([1, 10]))      # 定义第一层偏置，1行10列，全为0
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1     # 矩阵相乘，并加上偏置项  计算第一层线性组合
L1 = tf.nn.tanh(Wx_plus_b_L1)                          # 定义激活函数 使用的是tanh激活函数 激活函数的输出作为下一层的输入

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))     # 定义第二层权重，10行1列，随机生成
biases_L2 = tf.Variable(tf.zeros([1, 1]))               # 定义第二层偏置，1行1列，全为0
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2    # 矩阵相乘，并加上偏置项  计算第二层线性组合
prediction = tf.nn.tanh(Wx_plus_b_L2)                   # 定义激活函数 使用的是tanh激活函数 激活函数的输出作为下一层的输入

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y - prediction))        # 均方差函数，y-prediction为差值，square为平方，reduce_mean为求均值

# 定义反向传播方法 (使用梯度下降算法训练)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)    # 使用梯度下降优化器，学习率为0.1

# 创建 TensorFlow 会话
with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())  # 初始化所有变量

    for _ in range(2000):                    # 训练2000次
        sess.run(train_step, feed_dict={x: x_data, y: y_data})    #  运行训练步骤，传入训练数据, train_step为优化器，x,y为占位符，传入数据

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()                                                   # 创建一个图形实例，方便同时多图
    plt.scatter(x_data, y_data)                                    # 绘制散点图，显示真实值
    plt.plot(x_data, prediction_value, 'r-', lw=5)          # 绘制预测值曲线，红色实线，线宽为 5
    plt.show()                                                     # 显示图像

