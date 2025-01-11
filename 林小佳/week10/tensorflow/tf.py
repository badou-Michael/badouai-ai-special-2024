import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 随机生成数据集
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + noise

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 隐藏层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1     # 预激活函数
L1 = tf.nn.tanh(Wx_plus_b_L1)       # 此处不能选择sigmoid、loss太大

# 输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 反向传播
loss = tf.reduce_mean(tf.square(y - prediction))
train_step = tf.train.GradientDescentOptimizer(0.4).minimize(loss)

# 执行会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 进行训练
    for i in range(2000):
        res = sess.run([train_step, loss], feed_dict={x: x_data, y: y_data})
        print(res)      # 用fetch方法获取每次训练的loss
    # 开始推理
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
    plt.show()
