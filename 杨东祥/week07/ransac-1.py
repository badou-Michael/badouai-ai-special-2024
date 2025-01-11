import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

# 设置随机种子，确保每次结果一致
np.random.seed(48)

# 生产y=2x+1的线性数据
n_samples = 100
x = np.linspace(0, 10, n_samples)
y = 2 * x + 1

# 给y添加高斯噪声
y_noisy = y + np.random.normal(scale=2, size=n_samples)

# 添加10个不重复索引的点  离谱的调整这些点位的y值 变为局外点
n_outliers = 10
outliers_idx = np.random.choice(range(n_samples), size=n_outliers, replace=False)
y_noisy[outliers_idx] += 30

# 用最小二乘法拟合
linear_reg = LinearRegression()
linear_reg.fit(x.reshape(-1, 1), y_noisy)
# 预测回归拟合数据曲线
line_y_linear_reg = linear_reg.predict(x.reshape(-1, 1))


# 用ransac回归拟合
ransac = RANSACRegressor()
ransac.fit(x.reshape(-1, 1), y_noisy)
# 预测回归拟合数据曲线
line_y_ransac = ransac.predict(x.reshape(-1, 1))

plt.figure(figsize=(8,6))
# 显示噪声点
plt.scatter(x, y_noisy, label='Noisy', color='gray', alpha=0.5)
# 显示局外点
plt.scatter(x[outliers_idx], y_noisy[outliers_idx], color='orange', label='Outliers')

# 显示线性方程的曲线
plt.plot(x, y, label='y = 2x + 1', color='blue', linewidth=2)

# 显示带有噪声点局外点的数据经最小二乘法拟合的曲线
plt.plot(x, line_y_linear_reg, label='linear_reg', color='purple', linewidth=2)

# 显示带有噪声点局外点的数据经ransac拟合的曲线
plt.plot(x, line_y_ransac, label='ransac', color='red', linewidth=2)

plt.title('ransac linear regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.show()