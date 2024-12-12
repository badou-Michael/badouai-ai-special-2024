import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据生成
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 数据归一化（确保范围适合激活函数 tanh 的特性）
# x_data = (x_data - np.mean(x_data)) / np.std(x_data)
# y_data = (y_data - np.mean(y_data)) / np.std(y_data)

# 2. 定义网络参数
w_hidden = tf.Variable(tf.random.normal([1, 10], mean=0.0, stddev=1.0))  # 隐藏层权重
b_hidden = tf.Variable(tf.zeros([1, 10]))  # 隐藏层偏置
w_output = tf.Variable(tf.random.normal([10, 1], mean=0.0, stddev=1.0))  # 输出层权重
b_output = tf.Variable(tf.zeros([1, 1]))  # 输出层偏置

'''
权重初始化的重要性
权重的初始化决定了：

激活值的分布：初始权重的值会影响每一层神经元的激活值分布，进而影响梯度的大小。
梯度的大小：如果权重初始化不当，可能导致梯度消失或梯度爆炸问题。
网络的学习效率：权重初始化会影响网络从一开始是否能有效学习。
'''

'''
激活函数 tanh 的输出范围是 [-1, 1]，且具有如下特性：

当输入接近 0 时, tanh 的梯度较大，容易学习。
当输入较大或较小时（即输入远离 0 时) tanh 的梯度会接近 0, 导致梯度消失。
权重初始化的标准差决定了初始输入分布：

小的标准差 stddev (如默认值 0.1): 
隐藏层的输入值会较小，激活函数的输出主要处于线性区域。
这通常是稳定的，但对于复杂的非线性关系 (如 y = x^2), 初始权重的表达能力可能不足，学习速度较慢。

较大的标准差 stddev (如 1.0):
隐藏层的输入值分布较广, tanh 的非线性部分 (靠近 -1 或 1) 会被激活。
在这种情况下，网络初始状态的表达能力增强，拟合能力较强，但也可能导致梯度不稳定。
'''

# 学习率
lr = 0.1

# 3. 定义前向传播
def forward(x):
    """
    前向传播函数：计算网络的输出
    """
    hidden = tf.nn.tanh(tf.matmul(x, w_hidden) + b_hidden)  # 隐藏层计算
    output = tf.nn.tanh(tf.matmul(hidden, w_output) + b_output)  # 输出层计算
    return output

# 4. 训练
x_tensor = tf.convert_to_tensor(x_data, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y_data, dtype=tf.float32)

epochs = 2000
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = forward(x_tensor)
        loss = tf.reduce_mean(tf.square(y_tensor - y_pred))  # 均方误差损失

    # 计算梯度
    gradients = tape.gradient(loss, [w_hidden, b_hidden, w_output, b_output])

    # 更新权重和偏置
    w_hidden.assign_sub(lr * gradients[0])
    b_hidden.assign_sub(lr * gradients[1])
    w_output.assign_sub(lr * gradients[2])
    b_output.assign_sub(lr * gradients[3])

    # 打印损失值
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

# 5. 预测
y_pred = forward(x_tensor).numpy()

# 6. 可视化
plt.figure()
plt.scatter(x_data, y_data, label="True Data")  # 真实值散点图
plt.plot(x_data, y_pred, 'r-', lw=2, label="Predictions")  # 拟合结果曲线
plt.legend()
plt.title("TensorFlow 2: Aligned with TF1 Behavior")
plt.show()



# **TensorFlow 1 写法**
# 注释掉的部分展示了经典 TensorFlow 1 的实现方法
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
 

# 定义占位符
x = tf.placeholder(tf.float32, [None, 1])  # 输入数据占位符
y = tf.placeholder(tf.float32, [None, 1])  # 输出数据占位符

# 定义隐藏层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))  # 隐藏层权重矩阵
biases_L1 = tf.Variable(tf.zeros([1, 10]))  # 隐藏层偏置
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1  # 隐藏层线性组合
L1 = tf.nn.tanh(Wx_plus_b_L1)  # 隐藏层激活函数

# 定义输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))  # 输出层权重矩阵
biases_L2 = tf.Variable(tf.zeros([1, 1]))  # 输出层偏置
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2  # 输出层线性组合
prediction = tf.nn.tanh(Wx_plus_b_L2)  # 输出层激活函数

# 损失函数
loss = tf.reduce_mean(tf.square(y - prediction))  # 均方误差损失
# 优化器
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 梯度下降优化

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化所有变量
    for i in range(2000):  # 训练 2000 次
        sess.run(train_step, feed_dict={x: x_data, y: y_data})  # 前向传播 + 反向传播
        if i % 200 == 0:
            current_loss = sess.run(loss, feed_dict={x: x_data, y: y_data})
            print(f"Epoch {i}, Loss: {current_loss}")

    # 预测
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data, label="True Data")  # 散点是真实值
    plt.plot(x_data, prediction_value, 'r-', lw=2, label="Predictions")  # 曲线是预测值
    plt.legend()
    plt.title("TF1 Training Result")
    plt.show()
"""

