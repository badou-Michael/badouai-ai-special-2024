import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.disable_v2_behavior()

x_data = np.linspace(-0.5, 0.5, 200)[:, None]
# print(x_data.shape)

noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

x = tf.compat.v1.placeholder(tf.float32, [None, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 1])

w_1 = tf.Variable(tf.random.normal([1, 10]))
b_1 = tf.Variable(tf.zeros([1, 10]))
temp1 = tf.nn.tanh(tf.matmul(x, w_1) + b_1)

w_2 = tf.Variable(tf.random.normal([10, 1]))
b_2 = tf.Variable(tf.zeros([1, 1]))
output = tf.nn.tanh(tf.matmul(temp1, w_2) + b_2)

loss = tf.reduce_mean(tf.square(y - output))

train_step = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    output = sess.run(output, feed_dict={x: x_data})

    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, output, "r-", lw=5)  # 曲线是预测值
    plt.show()
