import numpy as np
import random
import matplotlib.pyplot as plt


def fit_line_least_squares(points):
    # 使用最小二乘法拟合直线
    A = np.vstack((points[:, 0], np.ones(len(points)))).T
    m, c = np.linalg.lstsq(A, points[:, 1], rcond=None)[0]
    return m, c


def calculate_residuals(points, m, c):
    # 计算残差
    predictions = m * points[:, 0] + c
    residuals = np.abs(points[:, 1] - predictions)
    return residuals


def ransac(points, iters, threshold, model_func, residual_func):
    # RANSAC算法
    best_fit = None
    best_inliers = 0

    for i in range(iters):
        # 随机选择两个点
        rand_idxs = random.sample(range(len(points)), 2)
        sample_points = points[rand_idxs, :]

        # 拟合模型
        fit = model_func(sample_points)

        # 计算所有点的残差
        residuals = residual_func(points, *fit)

        # 确定内点
        inliers = residuals < threshold

        # 如果内点数量大于当前最佳，则更新最佳拟合
        if np.sum(inliers) > best_inliers:
            best_inliers = np.sum(inliers)
            best_fit = fit

    return best_fit, best_inliers


# 示例数据
np.random.seed(0)
n_points = 100
n_outliers = 20
x = np.linspace(0, 10, n_points)
y = 2 * x + np.random.randn(n_points)  # 真实直线 y = 2x
outliers = np.random.uniform(-10, 10, (n_outliers, 2))  # 异常值
points = np.vstack((x, y)).T
points = np.vstack((points, outliers))

# RANSAC参数
iters = 1000
threshold = 1.0

# 运行RANSAC
best_fit, best_inliers = ransac(points, iters, threshold, fit_line_least_squares, calculate_residuals)

# 使用最小二乘法拟合直线
m_least_squares, c_least_squares = fit_line_least_squares(points)

# 绘制点、RANSAC拟合后的直线和最小二乘法拟合的直线
plt.figure(figsize=(10, 6))
plt.scatter(points[:, 0], points[:, 1], label='Data points')

if best_fit is not None:
    m, c = best_fit
    x_fit = np.linspace(min(points[:, 0]), max(points[:, 0]), 100)
    y_fit = m * x_fit + c
    plt.plot(x_fit, y_fit, color='red', label='RANSAC Fitted Line')

x_fit_least_squares = np.linspace(min(points[:, 0]), max(points[:, 0]), 100)
y_fit_least_squares = m_least_squares * x_fit_least_squares + c_least_squares
plt.plot(x_fit_least_squares, y_fit_least_squares, color='blue', label='Least Squares Fitted Line')

plt.legend()
plt.title('Line Fitting with RANSAC and Least Squares')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

print(f"RANSAC fit line: y = {m:.2f}x + {c:.2f}")
print(f"Least Squares fit line: y = {m_least_squares:.2f}x + {c_least_squares:.2f}")
print(f"Number of inliers: {best_inliers}")