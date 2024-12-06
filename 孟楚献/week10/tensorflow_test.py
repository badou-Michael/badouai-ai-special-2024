import matplotlib.pyplot as plt
import numpy as np
# 不能起同名.py文件
import tensorflow as tf

# 均分-0.5 - 0.5 200份
x_data = np.linspace(-0.5, 0.5, 200)[:,np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# placeholder存放数据   动态大小，n * 1
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 神经网络中间层
weights_l1 = tf.Variable(tf.random_normal((1, 10)))
biases_l1 = tf.Variable(tf.zeros([1, 10]))
wx_plus_b_l1 = tf.matmul(x, weights_l1) + biases_l1
l1 = tf.nn.tanh(wx_plus_b_l1)

print(biases_l1)
# 输出层  声明为Variable会随训练过程中调整，不然就不变
weights_l2 = tf.Variable(tf.random_normal((10, 1)))
biases_l2 = tf.Variable(tf.zeros((1, 1)))
wx_plus_b_l2 = tf.matmul(l1, weights_l2) + biases_l2
prediction = tf.nn.tanh(wx_plus_b_l2)

# 损失函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 反向传播算法
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})

    pridic_value = sess.run(prediction, feed_dict={x:x_data})

    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data,pridic_value,'b-',lw=5)   #曲线是预测值
    plt.show()
