import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def ransac(X, y, n_iterations=100, threshold=5, n_samples=2):
    best_inliers = []
    best_model = None

    # 在每次迭代中
    for _ in range(n_iterations):
        # 随机选择n_samples个点来拟合线性模型
        sample_indices = np.random.choice(len(X), size=n_samples, replace=False)
        X_sample = X[sample_indices]
        y_sample = y[sample_indices]

        # 拟合线性模型
        model = LinearRegression()
        model.fit(X_sample, y_sample)

        # 预测所有数据点的 y 值
        y_pred = model.predict(X)

        # 计算残差
        residuals = np.abs(y - y_pred)

        # 找出内点
        inliers = residuals < threshold
        num_inliers = np.sum(inliers)

        # 更新最佳模型
        if num_inliers > len(best_inliers):
            best_inliers = inliers
            best_model = model

    return best_model, best_inliers


# 生成示例数据
np.random.seed(0)
n_samples = 400
n_outliers = 100

# 生成线性数据
X = np.linspace(0, 10, n_samples).reshape(-1, 1)
y = 2 * X.ravel() + 1  # 正确模型 y = 2*x + 1

# 添加一些噪声
y += np.random.normal(size=n_samples)

# 添加离群点
X_outliers = np.random.uniform(0, 10, n_outliers).reshape(-1, 1)
y_outliers = np.random.uniform(10, 30, n_outliers)

# 将正常数据和离群点合并
X_combined = np.vstack((X, X_outliers))
y_combined = np.concatenate([y, y_outliers])

# 使用普通最小二乘法拟合
lin_reg = LinearRegression()
lin_reg.fit(X_combined, y_combined)
y_pred_lin_reg = lin_reg.predict(X_combined)

# 用户定义的参数
n_iterations = 100  # 迭代次数
n_samples = 10  # 随机选择点的数量

# 使用自定义 RANSAC 拟合
ransac_model, inliers = ransac(X_combined, y_combined, n_iterations=n_iterations, n_samples=n_samples)

# 预测 RANSAC 拟合的 y 值
y_pred_ransac = ransac_model.predict(X_combined)

# 绘图
plt.figure(figsize=(10, 6))

# 原始数据和离群点
plt.scatter(X, y, color='green', marker='o', label="Inliers (True Data)")
plt.scatter(X_outliers, y_outliers, color='red', marker='x', label="Outliers")

# 真实模型
plt.plot(X, 2 * X + 1, color="blue", linewidth=2, label="True Model (y = 2*x + 1)")

# 最小二乘法拟合
plt.plot(X_combined, y_pred_lin_reg, color="orange", linewidth=2, linestyle='--', label="Least Squares Fit")

# RANSAC 拟合
plt.plot(X_combined[inliers], y_pred_ransac[inliers], color="purple", linewidth=2, linestyle='-.', label="RANSAC Fit")

# 图例和标题
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Linear Fit, RANSAC Fit, and Least Squares Fit")
plt.show()
