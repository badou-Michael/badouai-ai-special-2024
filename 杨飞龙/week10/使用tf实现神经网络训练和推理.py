import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以获得可重复结果
np.random.seed(0)

# 使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# 生成一些噪声数据，均值为0，标准差为0.1
noise = np.random.normal(0, 0.5, x_data.shape)
# 计算x_data的线性关系，然后加上噪声，生成目标值y_data
y_data = 2 * x_data + noise

# 定义 TensorFlow 1.x 的输入和输出占位符
x = tf.placeholder(tf.float32, shape=[None, 1], name='x-input')
y = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')

# 定义模型的权重和偏置项
W = tf.Variable(tf.random_normal([1, 1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')

# 定义线性模型
prediction = tf.matmul(x, W) + b

# 定义损失函数
loss = tf.reduce_mean(tf.square(y - prediction))

# 定义优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss)

# 初始化全局变量
init = tf.global_variables_initializer()

# 运行训练
with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        sess.run(train_op, feed_dict={x: x_data, y: y_data})
        if i % 100 == 0:
            current_loss = sess.run(loss, feed_dict={x: x_data, y: y_data})
            print(f"Step {i}: Loss: {current_loss}")

    # 获取预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_data, prediction_value, 'r-', lw=5, label='Prediction')
    plt.legend()
    plt.show()
