import numpy as np


# 激活函数和其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# 初始化参数
np.random.seed(42)
input_size = 3  # 输入特征的数量
hidden_size = 4  # 隐藏层的神经元数
output_size = 1  # 输出层的神经元数

# 假设输入数据和标签
X = np.array([[0, 0, 1],
              [1, 0, 1],
              [0, 1, 1],
              [1, 1, 1]])

y = np.array([[0], [1], [1], [0]])  # XOR问题

# 初始化权重
W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

# 训练过程
learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    # 前向传播
    hidden_input = np.dot(X, W1)
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2)
    final_output = sigmoid(final_input)

    # 计算损失（MSE）
    loss = np.mean((y - final_output) ** 2)

    # 反向传播
    error_output = y - final_output
    d_output = error_output * sigmoid_derivative(final_output)

    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # 更新权重
    W2 += hidden_output.T.dot(d_output) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss}")

print("Final output after training:")
print(final_output)
