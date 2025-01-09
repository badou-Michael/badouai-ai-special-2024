import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 先准备一点数据， 输入，输出
x_date = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
print(x_date)
noise = np.random.normal(0, 0.02, x_date.shape)
y_date = np.square(x_date)+noise

# 把图上的节点准备好feed， 类型是浮点型， size是none，1
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 开始准备正向传播，计算隐藏层结果
wight_L1 = tf.Variable(tf.random.normal([1, 10]))  # 随机生成一个正态分布的行向量
biases_L1 = tf.Variable(tf.zeros([1, 10]))
wx_plus_b_L1 = tf.matmul(x, wight_L1)+biases_L1
# 这里是有200个样本，每一行都是一个隐藏层结果，偏置也会自动进行广播，正确的加入到每一行
L1 = tf.tanh(wx_plus_b_L1)   # 激活函数

# 计算输出层结果,先要生成矩阵和偏置量
wight_L2 = tf.Variable(tf.random.normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
wx_plus_b_L2 = tf.matmul(L1, wight_L2)+biases_L2
L2 = tf.nn.tanh(wx_plus_b_L2)

#计算损失函数
loss = tf.reduce_mean(tf.square(y-L2))
# 反向传播
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 开始执行图
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_date, y: y_date})
    predict = sess.run(L2, feed_dict={x: x_date})

# 画图
plt.figure()
plt.scatter(x_date, y_date)
plt.plot(x_date, predict, '-r',lw=4)
plt.show()
