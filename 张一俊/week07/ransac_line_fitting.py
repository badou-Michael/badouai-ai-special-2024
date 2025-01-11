import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

# 生成带噪声的数据
np.random.seed(0)
n_samples = 100
outliers_fraction = 0.3
n_outliers = int(outliers_fraction * n_samples)

# 正常的数据点：沿着一条直线
X = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
y = 0.5 * X.squeeze() + 0.1 * np.random.normal(size=n_samples)
# y = 0.5 * X.squeeze()

# 引入一些外群点（噪声点）
X[-n_outliers:] = 5 * np.random.rand(n_outliers, 1)
y[-n_outliers:] = 2 * np.random.rand(n_outliers)

# 使用 RANSAC 拟合数据
ransac = RANSACRegressor()
ransac.fit(X, y)

# 使用模型预测
line_X = np.arange(0, 5, 0.1)[:, np.newaxis]
line_y_ransac = ransac.predict(line_X)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='yellowgreen', marker='.')
plt.plot(line_X, line_y_ransac, color='blue')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('RANSAC line fitting(lubangxing)')
plt.show()
