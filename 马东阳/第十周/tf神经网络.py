# 导入模块
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成400个随机点
# 指定间隔起始点、终止端，以及指定分隔值总数（包括起始点和终止点）；最终函数返回间隔类均匀分布的数值序列。
# np.linspace(start = 0, stop = 100, num = 5)
# 使用 np.newaxis 增加一个新维度，变为二维数组；允许在数组的任意位置增加新的维度
x_data = np.linspace(-0.5, 0.5, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder存放输入数据
# placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
#   tf.placeholder( dtype,shape=None,name=None)
'''
dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
shape：数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
name：名称
'''
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 神经网络的中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))   # 形状为：连接上一个神经元个数*下一个神经元个数
biases_L1 = tf.Variable(tf.zeros([1, 10]) + 0.1)  # 加入偏置项
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1      # 表示 wx+b 的计算结果
L1 = tf.nn.tanh(Wx_plus_b_L1)  # 加入激活函数  tf.nn.relu()

# 输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))  # 形状为：连接上一个神经元个数*下一个神经元个数
biases_L2 = tf.Variable(tf.zeros([1, 1]) + 0.1)      # 加入偏置项
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2   # 表示 wx+b 的计算结果
prediction = tf.nn.tanh(Wx_plus_b_L2)  # 加入激活函数

# 损失函数：均方差函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 定义反向传播算法（使用梯度下降算法训练）, tf.train.AdamOptimizer(0.1).minimize(loss)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    init = tf.global_variables_initializer()
    sess.run(init)
    # 训练1000次
    for i in range(1000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})  #将x_data和y_data喂给tf变量x，y

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    # 散点，真实值
    # 曲线，预测值
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
