import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 启用 TensorFlow 1.x 的兼容模式
tf.compat.v1.disable_eager_execution()

# 生成一些真实数据（散点）
x = np.random.rand(100)
y = 3 * x + np.random.normal(0, 0.1, 100)  # 真实值 y = 3x + 噪声

# 创建 TensorFlow 图
X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1])
Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 1])

# 定义模型参数（权重和偏置） .reshape(1,1)是为了确保形状是[1,1]的二维张量，用于下面的tf.matmul(X, W)
W = tf.compat.v1.Variable(np.random.randn(1).reshape(1,1), dtype=tf.float32)
b = tf.compat.v1.Variable(np.random.randn(1), dtype=tf.float32)

# 定义模型输出（线性回归）
y_pred = tf.add(tf.matmul(X, W), b)

# 定义损失函数（均方误差）
loss = tf.reduce_mean(tf.square(y_pred - Y))

# 使用 Adam 优化器进行训练
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

# 使用 tf.Session() 运行图
with tf.compat.v1.Session() as sess:
    # 初始化变量
    sess.run(tf.compat.v1.global_variables_initializer())

    # 将数据转换为列向量
    x_train = x.reshape(-1, 1)
    y_train = y.reshape(-1, 1)

    # 训练模型
    for epoch in range(500):
        _, current_loss = sess.run([optimizer, loss], feed_dict={X: x_train, Y: y_train})

    # 预测结果
    y_pred_vals = sess.run(y_pred, feed_dict={X: x_train})

    # 绘制真实值（散点）和预测值（曲线）
    plt.scatter(x, y, color='blue', label='True Values (Scatter)', alpha=0.6)
    plt.plot(x, y_pred_vals, color='red', label='Predicted Values (Curve)', linewidth=2)
    plt.legend()
    plt.title('Scatter of True Values and Curve of Predicted Values')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
