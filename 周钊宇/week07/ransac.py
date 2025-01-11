import numpy as np
import matplotlib.pyplot as plt

def ransac(data, model_func, dist_func, n, k, t, debug=False):
    best_model = None
    best_consensus = []
    best_score = 0

    for i in range(k):
        indices = np.random.choice(len(data), n, replace=False)
        sample = np.array([data[i] for i in indices])

        # 估计模型参数
        model = model_func(sample)

        # 计算所有数据点到模型的距离
        distances = np.array([dist_func(point, model) for point in data])
        consensus = data[distances < t]

        # 如果当前模型的内点数量大于之前的最佳模型，则更新最佳模型
        if len(consensus) > best_score:
            best_model = model
            best_consensus = consensus
            best_score = len(consensus)

        if debug:
            print(f"Iteration {i+1}, model: {model}, inliers: {len(consensus)}")

    return best_model, best_consensus

# 模型函数：根据两个点拟合直线
def line_model_func(sample):
    x0, y0 = sample[0]
    x1, y1 = sample[1]
    a = (y1 - y0) / (x1 - x0) if x1 != x0 else np.inf
    b = y0 - a * x0
    return a, b

# 距离函数：计算点到直线的距离
def line_dist_func(point, model):
    a, b = model
    x, y = point
    return abs(a * x + b - y) / np.sqrt(a**2 + 1)

# 生成一些带有噪声的数据点
np.random.seed(0)
x = np.random.rand(100) * 100
y = 2 * x + np.random.randn(100) * 20  # 直线 y = 2x + noise

# 将数据点转换为 (x, y) 格式
data = np.column_stack((x, y))

# 应用RANSAC算法
model, consensus = ransac(data, line_model_func, line_dist_func, 2, 1000, 1)

# 绘制数据点和拟合的直线
plt.scatter(data[:, 0], data[:, 1], label='Data points')
plt.scatter(consensus[:, 0], consensus[:, 1], color='red', label='Consensus points')
x_vals = np.array([data[:, 0].min(), data[:, 0].max()])
y_vals = [model[0] * x + model[1] for x in x_vals]
plt.plot(x_vals, y_vals, label='Fitted line')
plt.legend()
plt.show()