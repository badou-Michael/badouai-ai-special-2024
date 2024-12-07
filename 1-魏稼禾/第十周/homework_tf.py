import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

x_data = np.linspace(-0.5, 0.5, 200)[:,np.newaxis]
noise = np.random.normal(0.0, 0.02, x_data.shape)
y_data = np.square(x_data)+noise

# 定义两个placeholder存放输入数据
x = tf.placeholder(tf.float32, [None,1])
y = tf.placeholder(tf.float32, [None,1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1,20]))
biases_L1 = tf.Variable(tf.zeros([1,20]))
Wx_plus_b_L1 = tf.matmul(x, Weights_L1)+biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)
# L1 = tf.nn.sigmoid(Wx_plus_b_L1)

# 定义输出层
Weights_L2 = tf.Variable(tf.random_normal([20,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2)+biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)
# prediction = tf.nn.sigmoid(Wx_plus_b_L2)
# prediction = Wx_plus_b_L2


loss = tf.reduce_mean(tf.square(y-prediction))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})
        
    prediction_value = sess.run(prediction, feed_dict={x:x_data})
    
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()