# Tensorflow—基本用法
# 1. 使用图 (graph) 来表示计算任务.
# 2. 在被称之为 会话 (Session) 的上下文 (context) 中执行图.
# 3. 使用 tensor 表示数据.
# 4. 通过 变量 (Variable) 维护状态.
# 5. 使用 feed 和 fetch 可以为任意的操作(arbitrary operation)赋值或者从其中获取数据。

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt


#使用numpy生成200个随机点, 构建训练集
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,x_data.shape)
y_data=np.square(x_data)+noise

# 定义图
#placeholder：占位符
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])

# 定义网络的中间层
weights_L1 = tf.Variable(tf.random_normal([1,10]))
bias_L1 = tf.Variable(tf.zeros([1,10]))
# 定义加权和操作
Wx_plus_b_L1 = tf.matmul(x,weights_L1) + bias_L1
# 定义激活函数操作
L1_output = tf.nn.tanh(Wx_plus_b_L1)

weights_L2 = tf.Variable(tf.random_normal([10,1]))
bias_L2 = tf.Variable(tf.zeros([1,1]))
# 定义加权和操作
Wx_plus_b_L2 = tf.matmul(L1_output,weights_L2) + bias_L2
# 定义激活函数操作
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 反向传播
# 定义损失函数（均方差）
loss_function = tf.reduce_mean(tf.square(y - prediction))
# 定义梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss_function)

with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 训练
    for i in range(2000):
        sess.run(train_step, feed_dict = {x:x_data, y:y_data})

    # 预测结果
    predict_value = sess.run(prediction, feed_dict = {x:x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, predict_value, 'r-', lw=5)  # 曲线是预测值
    plt.show()
