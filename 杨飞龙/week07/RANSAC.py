import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# 实现最小二乘法和RANSAC
# 生成随机数据
np.random.seed(1)  # 种子数为0代表序号0，每次生成的随机数是一样的0号
n_points = 1000   #200个数据
x = np.random.rand(n_points) * 20    #random.rand生成0-1的随机数，x是0-20的随机数
y = 3 * x + np.random.randn(n_points) * 2   # y 是根据线性关系 y = 3x 生成的，但加入了随机噪声（正态分布，标准差为2）

# 添加异常值
# outliers = np.random.rand(100, 2) *100    #生成100个随机的异常值点。生成一个20行2列的数组，数组中的元素是从[0.0, 1.0)区间内随机采样的浮点数
# y[outliers[:, 0].astype(int)] += 20   #将这些异常值添加到 y 数组中，使得某些数据点的 y 值异常地高

# 生成异常值的数量
n_outliers = 100

# 生成异常值的x偏移量
outlier_x_offsets = np.random.uniform(-5, 5, n_outliers) * np.random.rand(n_outliers) * 20

# 生成异常值的y偏移量，增加标准差以加大波动
outlier_y_offsets = np.random.normal(loc=0, scale=100, size=n_outliers)  # 增加标准差至15

# 打乱异常值的x和y偏移量的顺序
np.random.shuffle(outlier_x_offsets)
np.random.shuffle(outlier_y_offsets)

# 将打乱顺序的异常值添加到对应的x和y值上
outlier_indices = np.random.choice(n_points, n_outliers, replace=False)
x[outlier_indices] += outlier_x_offsets
y[outlier_indices] += outlier_y_offsets

# RANSAC 算法
def ransac(x, y, iterations=50, inlier_threshold=1.0):
    best_inliers_count = -1
    best_params = None
    for _ in range(iterations):
        idx = np.random.choice(len(x), 2, replace=False)
        model = np.polyfit(x[idx], y[idx], 1)
        inliers = np.abs(y - np.polyval(model, x)) < inlier_threshold
        inliers_count = np.sum(inliers)
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_params = model
    return best_params

# 最小二乘法
def residuals(p, x, y):
    return y - (p[0] * x + p[1])

def least_squares_method(x, y):
    p0 = [0, 0]
    plsq = least_squares(residuals, p0, args=(x, y))
    return plsq.x

# 运行 RANSAC 和最小二乘法
ransac_params = ransac(x, y)
lsq_params = least_squares_method(x, y)

# 绘制结果
plt.figure(figsize=(10, 5))
plt.scatter(x, y, label='Data Points')
plt.plot(x, ransac_params[0] * x + ransac_params[1], 'r-', label='RANSAC Line')
plt.plot(x, lsq_params[0] * x + lsq_params[1], 'b--', label='Least Squares Line')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('RANSAC vs Least Squares')
plt.show()
