import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 生成200个数据点，并加入噪声
x_data = np.linspace(-1, 1, 100)[:, np.newaxis]
noise = np.random.normal(0, 0.1, x_data.shape)
# y_data为目标值
y_data = np.square(x_data) - 0.5 + noise

# 定义两个placeholder存放输入和输出
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义一个简单的线性模型
W = tf.Variable(tf.random_normal([1, 10]))  # 权重
b = tf.Variable(tf.zeros([1, 10]))  # 偏量 也就是bias
# 进行wx+b的线性变换
wx_plus_b = tf.matmul(x, W) + b
# 添加激活函数tanh
y_pred = tf.nn.tanh(wx_plus_b)

# 设置输出层
W2 = tf.Variable(tf.random_normal([10, 1]))  # 权重
b2 = tf.Variable(tf.zeros([1, 1]))  # 偏量 也就是bias
# 进行wx+b的线性变换
wx_plus_b2 = tf.matmul(y_pred, W2) + b2
# 添加激活函数tanh
y_pred2 = tf.nn.tanh(wx_plus_b2)

# 定义损失函数
loss = tf.reduce_mean(tf.square(y - y_pred2))
# 定义反向传播算法
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_op, feed_dict={x: x_data, y: y_data})

    # 预测
    y_pred_value = sess.run(y_pred2, feed_dict={x: x_data})
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, y_pred_value, 'r-', lw=5)
    plt.show()
