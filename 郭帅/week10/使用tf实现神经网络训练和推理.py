import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
1、随机生成20个点
np.linspace(start, stop, num) 是 NumPy 中用于生成指定数量 等间距数值 的函数
[:, np.newaxis] 将其转换为 二维数组，变成了一个 列向量
np.random.normal 是 NumPy 中生成 正态分布随机数 的函数
np.square 是 NumPy 中的一个函数，用于 计算每个元素的平方
astype(np.float32) 来确保与 TensorFlow 的默认类型（tf.float32）一致
'''
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis].astype(np.float32)
noise = np.random.normal(0,0.02,x_data.shape).astype(np.float32)
y_data = np.square(x_data) + noise

'''
2、定义两个Variable用来存储数据
tf.Variable 是 TensorFlow 中用于 创建和管理可训练变量 的一个重要类
'''
x = tf.Variable(x_data,dtype=tf.float32)
y = tf.Variable(y_data,dtype=tf.float32)

'''
3、定义神经网络中间层
tf.matmul 是 TensorFlow 中用于执行 矩阵乘法 的函数
tf.nn.tanh 是 TensorFlow 中的一个 激活函数
'''
Weights_L1=tf.Variable(tf.random.normal([1,10],dtype=tf.float32))   # 生成权重矩阵（输入层-隐藏层）
biases_L1=tf.Variable(tf.zeros([1,10],dtype=tf.float32))            # 生成偏置项（截距）
Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1     # 隐藏层输入：y = xw + b
L1=tf.nn.tanh(Wx_plus_b_L1)                        # 隐藏层输出：经过激活函数的值

'''
4、定义神经网络输出层
'''
Weights_L2=tf.Variable(tf.random.normal([10,1],dtype=tf.float32))
biases_L2=tf.Variable(tf.zeros([1,1],dtype=tf.float32))
Wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2
prediction=tf.nn.tanh(Wx_plus_b_L2)

'''
5、定义损失函数（均方差函数）
tf.square 用于 计算每个元素的平方
tf.reduce_mean(x) 它会计算张量 x 中所有元素的 平均值
loss是计算过程中的一个中间值，可以在每次迭代中直接计算，所以这步可以删掉
'''
# loss=tf.reduce_mean(tf.square(y - prediction))

'''
6、定义反向传播算法（使用梯度下降算法训练）
tf.optimizers.SGD(0.1) 创建了一个 随机梯度下降优化器，学习率为 0.1
'''
optimizer = tf.optimizers.SGD(0.1)

'''
7、创建训练过程
tf.GradientTape 是 TensorFlow 中用于自动计算梯度的工具，它会记录 tape 作用域内的计算过程，以便之后计算梯度并进行反向传播
tape.gradient(...)：计算的是损失函数对每个参数的偏导数，也就是梯度
optimizer.apply_gradients(...)：通过优化器（此处是随机梯度下降 SGD）应用梯度来更新神经网络的参数（权重和偏置）
zip(grads, [...])：zip 函数将计算得到的梯度 grads 和待更新的参数（权重和偏置）打包成一对对元组，并传递给优化器
【新权重 = 旧权重 - 学习率 * 偏导数（损失/权重）】
'''
epochs = 2000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        # 计算预测值和损失
        prediction_value = tf.nn.tanh(
            tf.matmul(tf.nn.tanh(tf.matmul(x, Weights_L1) + biases_L1), Weights_L2) + biases_L2)  # 前向传播
        loss_value = tf.reduce_mean(tf.square(y - prediction_value))  # 计算损失

    # 计算梯度并应用梯度更新参数
    grads = tape.gradient(loss_value, [Weights_L1, biases_L1, Weights_L2, biases_L2])  # 计算梯度
    optimizer.apply_gradients(zip(grads, [Weights_L1, biases_L1, Weights_L2, biases_L2]))  # 更新权重和偏置

    if epoch % 100 == 0:  # 每100次打印一次损失
        print(f"Epoch {epoch}, Loss: {loss_value.numpy()}")

'''
8、根据上面最终的权重值，获得预测值
'''
prediction_value = tf.nn.tanh(
    tf.matmul(tf.nn.tanh(tf.matmul(x_data, Weights_L1) + biases_L1), Weights_L2) + biases_L2).numpy()

'''
9、绘制图像
'''
plt.figure()
plt.scatter(x_data, y_data)  # 散点图是实际数据
plt.plot(x_data, prediction_value, 'r-', lw=5)  # 曲线是预测值
plt.show()
