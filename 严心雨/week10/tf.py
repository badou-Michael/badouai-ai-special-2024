import numpy
import tensorflow as tf
import matplotlib.pyplot as plt

#使用numpy生成200个随机点
"""
np.linspace(start,stop,num) 通过定义均匀间隔创建数值序列
numpy.random.normal(mu,sigma,n):均值，标准差，输出形状大小
np.newaxis：在数组中增加一个新的维度，使得一维数组变成二维数组，二维数组变成三维数组
y_data：自己设的正确答案
希望结果：希望经过训练之后推理函数也能给一个值就能得到一个很好的y_data
"""
x_data = numpy.linspace(-0.5,0.5,200)[:,numpy.newaxis]
noise = numpy.random.normal(0,0.02,x_data.shape)
y_data = numpy.square(x_data)+noise

#定义两个placeholder-占位符存放输入数据
"""
y：标签
[None,1]:形状 代表列是1，行不定
"""
x = tf.compat.v1.placeholder(tf.float32,[None,1])
y = tf.compat.v1.placeholder(tf.float32,[None,1])

#定义神经网络隐藏层1
"""
tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)：从服从指定正太分布的序列中随机取出指定个数的值
"""
[1]#权重初始化
weight_l1 = tf.Variable(tf.random.normal([1,10])) #1*10
[2]#偏置
bias_l1 = tf.Variable(tf.zeros([1,10]))
[3]#y=wx+b
Wx_plus_b_l1 = tf.matmul(x,weight_l1)+bias_l1# (N*1) * (1*10)
[4]#激活函数
L1 = tf.nn.tanh(Wx_plus_b_l1)#N*10
print(L1.shape)
#定义神经网络隐藏层2
[1]#权重初始化
weight_l2 = tf.Variable(tf.random.normal([10,1])) #10*1
print(weight_l2.shape)
[2]#偏置
"""
tf.zeros 默认使用 float32
numpy.zeros 默认使用 float64
"""
bias_l2 = tf.Variable(tf.zeros([1,1]))#这里不能写tf,zeros[10,1],因为L1.shape = [?,10],bias_l2的shape应为[?,1],所以不能写明[10,1]
print(bias_l2.shape)
[3]#y=wx+b
Wx_plus_b_l2 = tf.matmul(L1,weight_l2)+bias_l2 # (N*10) * (10*1)
[4]#激活函数
Prediction = tf.nn.tanh(Wx_plus_b_l2)# N*1

#定义损失函数（均方差函数MSE）
"""
tf.reduce_mean函数用于计算张量tensor沿着指定的数轴(tensor的某一维度)上的平均值，主要用作降维或者计算tensor(图像)的平均值
"""
loss = tf.reduce_mean(tf.square(y-Prediction))
#定义反向传播算法（自动微分工具 使用梯度下降算法训练使loss最小化）
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

with tf.Session() as sess:
    #变量初始化
    """
    因为前面有variable，所以先初始化这些变量
    """
    sess.run(tf.global_variables_initializer())
    #训练2000次 epoch=2000
    for i in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    #获得预测值
    """
    这里不严谨，应该用另外的数据而不是用于训练的数据，但是这儿是为了方便
    """
    prediction_value = sess.run(Prediction,feed_dict={x:x_data})

    #画图
    """
    plt.figure 用于创建一个新的图形或激活一个已经存在的图像
    """
    plt.figure()
    plt.scatter(x_data,y_data)#真实值
    plt.plot(x_data,prediction_value,'r-',lw=5)#曲线是预测值
    plt.show()
