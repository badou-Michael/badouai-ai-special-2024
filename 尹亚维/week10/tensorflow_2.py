import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy随机生成200个点并且给随机点加上噪音，转换成float32 类型
# 1.使用 numpy 生成了 200 个在区间 [-0.5, 0.5] 内均匀分布的随机点 x_data
# 2.为每个 x_data 添加了正态分布的噪声，并生成了目标值y_data
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis].astype(np.float32)
noise = np.random.normal(0, 0.02, x_data.shape).astype(np.float32)
y_data = np.square(x_data) + noise

# 定义神经网络中间层
# 生成一个形状为 [1, 10] 的张量，其中的元素是从正态分布（均值为 0，标准差为 1）中随机抽取的浮点数（float32 类型）
# 生成的张量是一个二维数组，具有 1 行和 10 列
Weights_L1 = tf.Variable(tf.random.normal([1, 10], dtype=tf.float32))
biases_L1 = tf.Variable(tf.zeros([1, 10], dtype=tf.float32))  # 加入偏置项
Wx_plus_b_L1 = tf.matmul(x_data, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)  # 加入激活函数

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random.normal([10, 1], dtype=tf.float32))
biases_L2 = tf.Variable(tf.zeros([1, 1], dtype=tf.float32))  # 加入偏置项
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)  # 加入激活函数

# 定义损失函数(均方差函数)
# 计算每个样本的真实值 y_data 与预测值 prediction 之间的差值
loss = tf.reduce_mean(tf.square(prediction - y_data))

# 定义反向传播算法
optimizer = tf.optimizers.SGD(learning_rate=0.1)

# 训练2000次
# tf.GradientTape 记录前向传播的操作，以便后续计算梯度
# 使用 tape.gradient 计算损失相对于各个变量的梯度
# 使用 optimizer.apply_gradients 应用这些梯度来更新模型参数
for i in range(2000):
    with tf.GradientTape() as tape:
        Wx_plus_b_L1 = tf.matmul(x_data, Weights_L1) + biases_L1
        L1 = tf.nn.tanh(Wx_plus_b_L1)
        Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
        prediction = tf.nn.tanh(Wx_plus_b_L2)
        loss = tf.reduce_mean(tf.square(prediction - y_data))
    gradients = tape.gradient(loss, [Weights_L1, biases_L1, Weights_L2, biases_L2])
    optimizer.apply_gradients(zip(gradients, [Weights_L1, biases_L1, Weights_L2, biases_L2]))

# 获得预测值
# 确保每次迭代时都能使用最新的权重和偏置来计算预测值和损失
Wx_plus_b_L1 = tf.matmul(x_data, Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 画图
plt.figure()
plt.scatter(x_data, y_data)  # 散点是真实值
plt.plot(x_data, prediction, 'r-', lw=5)  # 曲线是预测值
plt.show()