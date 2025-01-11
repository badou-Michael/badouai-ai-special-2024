import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  # 输入数据，范围从-0.5到0.5，200个点
noise = np.random.normal(0, 0.02, x_data.shape)  # 正态分布噪声
y_data = np.square(x_data) + noise  # 输出数据是x的平方，加上噪声

# 使用Keras的Sequential API定义神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='tanh', input_shape=(1,)),  # 隐藏层，10个神经元，激活函数为tanh
    tf.keras.layers.Dense(1, activation='tanh')  # 输出层，1个神经元，激活函数为tanh
])

# 编译模型，使用均方误差损失函数，优化器为梯度下降优化器
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='mean_squared_error')

# 训练模型，训练2000次
history = model.fit(x_data, y_data, epochs=2000, verbose=0)

# 预测数据
prediction_value = model.predict(x_data)

# 画图
plt.figure()
plt.scatter(x_data, y_data)  # 散点图表示真实数据
plt.plot(x_data, prediction_value, 'r-', lw=5)  # 红色曲线表示预测值
plt.show()
