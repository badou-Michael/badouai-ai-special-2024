import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 使用 numpy 生成 200 个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# np.linspace(-0.5, 0.5, 200) 创建了 200 个在 -0.5 到 0.5 之间均匀分布的值，
# [:, np.newaxis] 将其转换为列向量，每个值作为一个样本。

noise = np.random.normal(0, 0.02, x_data.shape)
# 使用 numpy 的 random.normal 函数生成均值为 0、标准差为 0.02 的正态分布噪声，形状与 x_data 相同。

y_data = np.square(x_data) + noise
# 根据函数 y = x^2 添加噪声生成目标数据 y_data。

# 定义两个 placeholder 存放输入数据
x = tf.placeholder(tf.float32, [None, 1])
# 定义一个占位符 x，用于接收输入数据，形状为 [任意数量的样本, 1]，表示每个样本是一个一维数据。

y = tf.placeholder(tf.float32, [None, 1])
# 定义一个占位符 y，用于接收目标数据，形状与 x 相同。

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
# 创建一个形状为 [1, 10] 的随机正态分布的权重变量，用于中间层。

biases_L1 = tf.Variable(tf.zeros([1, 10]))
# 创建一个形状为 [1, 10] 的全零偏置变量，用于中间层。

Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
# 计算中间层的线性组合，即输入 x 与权重相乘再加上偏置。

L1 = tf.nn.tanh(Wx_plus_b_L1)
# 对中间层的线性组合应用双曲正切激活函数，引入非线性。

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
# 创建一个形状为 [10, 1] 的随机正态分布的权重变量，用于输出层。

biases_L2 = tf.Variable(tf.zeros([1, 1]))
# 创建一个形状为 [1, 1] 的全零偏置变量，用于输出层。

Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
# 计算输出层的线性组合，即中间层的输出 L1 与权重相乘再加上偏置。

prediction = tf.nn.tanh(Wx_plus_b_L2)
# 对输出层的线性组合应用双曲正切激活函数，得到最终的预测结果。

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y - prediction))
# 计算预测值与目标值之间的均方误差作为损失函数，tf.square 计算平方差，tf.reduce_mean 计算平均值。

# 定义反向传播算法（使用梯度下降算法训练）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# 创建一个梯度下降优化器，学习率为 0.1，用于最小化损失函数。

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 初始化所有的 TensorFlow 变量。

    # 训练 2000 次
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    # 在每次迭代中，将输入数据 x_data 和目标数据 y_data 传入，运行训练步骤 train_step，更新模型参数。

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 使用训练好的模型对输入数据 x_data 进行预测，得到预测值 prediction_value。

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)
    # 绘制真实数据的散点图，x_data 是输入，y_data 是目标值。

    plt.plot(x_data, prediction_value, 'r-', lw=5)
    # 绘制预测值的曲线，颜色为红色，线宽为 5。

    plt.show()
    # 显示图形。
