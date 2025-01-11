import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点
# np.linspace表示生成-0.5到0.5之间，200个随机数，一维
# np.newaxis在数组的指定位置增加一个新的轴（维度）生成
# 变成二维也可以是  data_x = np.array(data_x,ndmin=2).T
data_x = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# loc正太分布均值，scale 标准差，size形状，生成（200，1）的正太分布的噪声
noise = np.random.normal(0, 0.02, data_x.shape)
# y = x**2+noise
y_data = np.square(data_x) + noise
# 定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
w_l1 = tf.Variable(tf.random_normal([1, 10]))
b1 = tf.Variable(tf.zeros([1, 10]))
# y = wx+b,隐藏层输入结果 (200,1)x(1,10) = (200,10)
hinput = tf.matmul(x, w_l1) + b1
# 隐藏层输入结果添加激活翰苏得到输出结果，
houtput = tf.nn.tanh(hinput)

# 定义神经网络输出层
w_l2 = tf.Variable(tf.random_normal([10, 1]))
b2 = tf.Variable(tf.zeros([1, 1]))
finalinput = tf.matmul(houtput, w_l2) + b2
finaloutput = tf.nn.tanh(finalinput)

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y - finaloutput))
# 定义反向传播算法（使用梯度下降算法训练）
trainstep = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 循环进行训练
    for i in range(2000):
        sess.run(trainstep, feed_dict={x: data_x, y: y_data})

    # 推理获取预测值
    predict = sess.run(finaloutput, feed_dict={x: data_x})
    # 画图
    plt.figure()
    plt.scatter(data_x, y_data)
    plt.plot(data_x, predict, 'r-', lw=5)  # 曲线是预测值
    plt.show()
