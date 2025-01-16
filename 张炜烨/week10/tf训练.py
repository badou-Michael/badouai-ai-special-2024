import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def generate_data():
    """
    生成训练数据：y = x^2 + 噪声
    """
    # 生成 200 个点，x 的取值范围在 [-0.5, 0.5]
    x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    # 添加噪声
    noise = np.random.normal(0, 0.02, x_data.shape)
    y_data = np.square(x_data) + noise
    return x_data, y_data


def build_network(x):
    """
    构建一个简单的两层神经网络。
    输入层 -> 隐藏层（10个神经元，tanh激活） -> 输出层（1个神经元，tanh激活）
    """
    # 隐藏层：1 -> 10
    weights_l1 = tf.Variable(tf.random_normal([1, 10]))
    biases_l1 = tf.Variable(tf.zeros([1, 10]))
    wx_plus_b_l1 = tf.matmul(x, weights_l1) + biases_l1
    hidden_layer = tf.nn.tanh(wx_plus_b_l1)  # 激活函数

    # 输出层：10 -> 1
    weights_l2 = tf.Variable(tf.random_normal([10, 1]))
    biases_l2 = tf.Variable(tf.zeros([1, 1]))
    wx_plus_b_l2 = tf.matmul(hidden_layer, weights_l2) + biases_l2
    output_layer = tf.nn.tanh(wx_plus_b_l2)  # 激活函数

    return output_layer


def train_network(x_data, y_data, epochs=2000, learning_rate=0.1):
    """
    训练神经网络并返回预测结果。
    """
    # 定义占位符（输入数据和目标值）
    x = tf.compat.v1.placeholder(tf.float32, [None, 1], name="x")
    y = tf.compat.v1.placeholder(tf.float32, [None, 1], name="y")

    # 构建网络
    prediction = build_network(x)

    # 定义损失函数（均方误差）
    loss = tf.reduce_mean(tf.square(y - prediction))

    # 定义优化器（梯度下降法）
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # 启动会话训练
    with tf.compat.v1.Session() as sess:
        # 初始化变量
        sess.run(tf.compat.v1.global_variables_initializer())

        # 训练指定轮次
        for epoch in range(epochs):
            _, current_loss = sess.run([optimizer, loss], feed_dict={x: x_data, y: y_data})

            # 每 200 次打印一次损失
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {current_loss:.6f}")

        # 获取最终的预测值
        prediction_value = sess.run(prediction, feed_dict={x: x_data})

    return x_data, y_data, prediction_value


def visualize_results(x_data, y_data, prediction_value):
    """
    可视化真实值和预测值。
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, label="Real Data", s=10)  # 散点图（真实值）
    plt.plot(x_data, prediction_value, 'r-', lw=2, label="Prediction")  # 曲线（预测值）
    plt.legend()
    plt.title("Neural Network Fit for y = x^2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # 1. 生成数据
    x_data, y_data = generate_data()

    # 2. 训练网络
    x_data, y_data, prediction_value = train_network(x_data, y_data)

    # 3. 可视化结果
    visualize_results(x_data, y_data, prediction_value)