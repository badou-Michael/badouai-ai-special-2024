import numpy as np
import random

def fit_line(points):
    """使用最小二乘法拟合直线"""
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    coefficients = np.linalg.lstsq(A, y, rcond=None)[0]
    return coefficients

def calculate_residuals(points, coefficients):
    """计算残差"""
    x = points[:, 0]
    predictions = coefficients[0] * x + coefficients[1]
    residuals = np.abs(predictions - points[:, 1])
    return residuals

def ransac(points, num_iterations=100, inlier_threshold=1.0):
    """RANSAC算法实现"""
    best_inliers_count = -1
    best_coefficients = None

    for i in range(num_iterations):
        # 随机选择两个点
        rand_idxs = random.sample(range(len(points)), 2)
        sample_points = points[rand_idxs, :]
        
        # 拟合直线
        coefficients = fit_line(sample_points)
        
        # 计算所有点的残差
        residuals = calculate_residuals(points, coefficients)
        
        # 识别内点
        inliers = residuals < inlier_threshold
        inliers_count = np.sum(inliers)
        
        # 更新最佳模型
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_coefficients = coefficients

    return best_coefficients, best_inliers_count

# 生成一些数据
np.random.seed(42)
num_points = 100
x = np.random.rand(num_points) * 10
y = 3 * x + np.random.randn(num_points) * 2  # 直线 y = 3x + N(0, 2)

# 添加一些异常值
x = np.append(x, [10, 20, 30, 40])
y = np.append(y, [80, 100, 120, 140])  # 异常值

points = np.column_stack((x, y))

# 运行RANSAC
best_coefficients, best_inliers_count = ransac(points)
print(f"Best model: y = {best_coefficients[0]:.2f}x + {best_coefficients[1]:.2f}")
print(f"Number of inliers: {best_inliers_count}")
