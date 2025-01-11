import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用 numpy 生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  # linspace 用于生成指定范围内的等间隔点  newaxis 是一个特殊的索引工具，用来增加数组的维度
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise  # y_data = x_data**2 + noise

# 定义两个placeholder存放数据
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
Weight_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weight_L1)+biases_L1
'''
nn 是 TensorFlow 中 neural network（神经网络）模块 的缩写
这个模块包含了与神经网络相关的操作，比如激活函数、卷积、池化等操作

'''
L1 = tf.nn.tanh(Wx_plus_b_L1) # 加入激活函数

# 定义神经网络输出层
Weight_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2=tf.matmul(L1,Weight_L2)+biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 定义损失函数（均方差函数）
loss = tf.reduce_mean(tf.square(y-prediction)) # 采用MSE
# 定义反向传播算法（使用梯度下降算法训练）

prediction = tf.nn.tanh(Wx_plus_b_L2)
'''
表示使用 梯度下降优化器（Gradient Descent Optimizer），学习率设为 0.1
'''
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 自动微分

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 训练2000次
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})

    # 获取预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data,y_data)
    '''
    r b g k：红色 蓝色 绿色 黑色
    ':'：点线  '--'：虚线  '-.'：点划线
    lw指线条宽度linewidth，默认值是1
    '''
    plt.plot(x_data,prediction_value,'r-',lw = 10)
    plt.show()

# # 生成一个具有写权限的日志文件操作对象，将当前命名空间的计算图写进日志中
# writer = tf.summary.FileWriter('logs',tf.get_default_graph())
# writer.close()
