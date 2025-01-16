import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
tf.disable_v2_behavior()  # 目的是为了禁用v2版本的急切执行

def tflianxi():
    # 创建一个张量
    ts1 = tf.constant([1, 2, 3])
    ts2 = tf.constant(5)
    ts21 = tf.constant(5, dtype=tf.float32)
    ts3 = tf.Variable(0, name='var3')
    print(ts1)  # 这打印的是类
    print(ts2)
    print(ts3)

    # 用张量进行计算
    op1 = tf.add(ts1, ts2)  # 实行加法运算
    op2 = ts3.assign(ts2)  # 这是tf2的代码，tf1的是tf.assign(ts3, ts2)，把ts2的值赋给ts3

    # 创建几个占位符
    em1 = tf.placeholder(tf.float32)  # 代表变量em1现在临时用一个空的32位浮点数占位，以待后续赋值
    em2 = tf.placeholder(tf.float32)
    op3 = tf.multiply(em1, em2)  # multiply是乘法

    # 初始化所有全局变量
    initvar = tf.global_variables_initializer()  # 这是tf1的代码

    # 启动容器
    # with tf.Session() as sess:  # tf1代码，tf2不再需要显式执行Session
    with tf.Session() as sess:  # tf2中使用图模式，调用tf.function的Session
        sess.run(initvar)
        print(sess.run(op1))
        print(sess.run(op2))
        print(sess.run(ts1))
        print(sess.run([ts2, ts3]))  # 可以使用序列的形式将两个变量在一个run中输出，人们管这个叫fetch操作
        print(sess.run((ts2, ts3)))
        print(sess.run(ts3))
        print(sess.run(tf.add(op1, op2)))
        print('qqqqq', sess.run(op3, feed_dict={em1: [1, 2, 3], em2: [5, 6, 7]}))  # 通过在run方法中使用feed_dict参数可以实现在run的时候对当时定义的空变量赋值
        print(sess.run(em1 * ts21, feed_dict={em1: 3., ts21: 2.}))  # 还可以对已经有值的变量进行再赋值
        print(sess.run(tf.add(tf.constant([5, 2, 3]), tf.constant([1, 2, 1]))))


# 使用numpy生成训练集
src_x = np.linspace(-1, 1, 300)[:, np.newaxis]  # linspace可以按照数量要求创建等间距的数组
# print(src_x)
# print(src_x.shape)
src_noise = np.random.normal(0, 0.05, src_x.shape)
# print(src_noise)
# print(src_noise.shape)
src_y = np.square(src_x) + src_noise  # square是数组内元素求平方操作


# 生成测试集
test_x = np.linspace(-1, 1, 500)[:, np.newaxis]
# print(test_x)
# print(test_x.shape)


# 定义两个空的变量存数据
x = tf.placeholder(tf.float32, [None, 1])  # 创建一个能保存n行1列的变量
y = tf.placeholder(tf.float32, [None, 1])


# 定义神经网络的中间层
Wmid = tf.Variable(tf.random.normal([1, 10]))  # 设定w
bmid = tf.Variable(tf.zeros([1, 10]))  # 设定bias
Zmid = tf.matmul(x, Wmid) + bmid  # 做wx+b的公式计算
Amid = tf.nn.tanh(Zmid)  # 用双曲正切函数作为激活函数


# 定义神经网络的输出层
Wout = tf.Variable(tf.random.normal([10, 1]))
bout = tf.Variable(tf.zeros([1, 1]))
Zout = tf.matmul(Amid, Wout) + bout
# Aout = tf.nn.sigmoid(Zout)  # 用sigmoid作为激活函数
Aout = tf.nn.tanh(Zout)  # 用双曲正切函数作为激活函数


# 定义损失函数
loss = tf.reduce_mean(tf.square(y - Aout))  # 损失函数用均方差，即均reduce_mean，方tf.square，差y-Aout


# 定义反向传播算法
backprop = tf.gradients(loss, Wout)
backprop = tf.train.GradientDescentOptimizer(0.2).minimize(loss)


# 初始化变量
initvar = tf.global_variables_initializer()


# 开始训练，定义容器
with tf.Session() as sess:
    sess.run(initvar)  # 执行初始化变量

    # 开始训练，定义循环为2000次
    for _ in range(2000):
        sess.run(backprop, feed_dict={x: src_x, y: src_y})

    # 进行测试
    pred = sess.run(Aout, feed_dict={x: test_x})

    # 通过plt画图
    plt.scatter(src_x, src_y)
    plt.plot(test_x, pred, 'r-', lw=5)
    plt.show()


















