import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt

# 生成随机数据
X = np.linspace(-0.5, 0.5, 1000)[:,np.newaxis]
Y = X ** 3 + np.square(X) + np.random.normal(0, 0.03, X.shape)  # 真实值加上一些噪声

# 定义输入和输出的placeholder
X_placeholder = tf.placeholder(tf.float32, [None, 1])
Y_placeholder = tf.placeholder(tf.float32, [None, 1])


# 定义模型结构(隐藏层)
W1 = tf.Variable(tf.random_normal([1, 20]), name='weight1')
b1 = tf.Variable(tf.random_normal([1, 20]), name='bias1')
hidden_output = tf.nn.tanh(tf.matmul(X_placeholder, W1) + b1)

# 定义模型结构(输出层)
W2 = tf.Variable(tf.random_normal([20, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1, 1]), name='bias2')
Y_pred = tf.nn.tanh(tf.matmul(hidden_output, W2) + b2)

# 定义损失函数
loss = tf.reduce_mean(tf.square(Y_placeholder - Y_pred))

# 定义optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(loss)


# 启动会话
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())

    # 训练模型
    for step in range(4000):
        sess.run(optimizer, feed_dict={X_placeholder: X, Y_placeholder: Y})

    # 评估模型
    prediction_value = sess.run(Y_pred,feed_dict={X_placeholder: X})

    #画图
    plt.figure()
    plt.scatter(X,Y)
    plt.plot(X,prediction_value, 'k-', lw=6)
    plt.show()
