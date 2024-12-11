#ransac算法

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
plt.rcParams['font.family'] = 'SimHei'  # 设定字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 1. 生成200个数据点，假设真实线性方程 y = 2 * x + 1
np.random.seed(0)  # 设置随机种子，保证每次生成相同数据,加入噪声是为了模拟现实中数据中常见的随机波动或误差，使得数据更具有现实性
X = np.random.rand(200, 1) * 10  # 随机生成 200 个 X 值, np.random.rand 用于生成均匀分布的随机数，值的范围是 [0, 1)
y = 2 * X + 1 +  np.random.randn(200, 1)  # y = 2x + 1 + 噪声, np.random.randn() 用于生成标准正态分布的随机数，均值为0，标准差为1。

# 2. 添加一些噪声点（局外点）
n_outliers = 30  # 设置30个局外点
X_outliers = np.random.rand(n_outliers, 1) * 10
y_outliers = np.random.rand(n_outliers, 1) * 20  # 这些点的y值是随机的

# 将噪声点添加到原始数据中
X_all = np.vstack([X, X_outliers])
y_all = np.vstack([y, y_outliers])

# 3. 使用 RANSAC 算法来拟合线性模型
ransac = RANSACRegressor()  # RANSACRegressor() 是一个类（类对象），而 .fit() 是该类中的一个方法
ransac.fit(X_all, y_all)    # 需要先创建类的实例（即对象），然后才能调用该实例的方法

# 4. 获取 RANSAC 估计的线性模型参数
slope = ransac.estimator_.coef_[0, 0]  # 提取斜率值,[0, 0] 用于访问 二维数组 中的元素,第一个行、第一列的元素
intercept = ransac.estimator_.intercept_[0]  # [0] 提取截距值,用于访问 一维数组 中的元素

# 输出 RANSAC 拟合的结果
print(f"拟合的线性模型: y = {slope:.2f} * x + {intercept:.2f}")   # :.2f 是格式化字符串，表示将变量 slope 和 intercept 的值格式化为小数点后两位的浮动数字。

# 5. 绘制结果
plt.figure(figsize=(8, 6))

# 绘制所有数据点（包括噪声点）
plt.scatter(X_all, y_all, color='gray', alpha=0.5, label='数据点')  # alpha=0.5 透明度，半透明

# 绘制 RANSAC 拟合的线
line_X = np.linspace(0, 10, 1000).reshape(-1, 1)
line_y = ransac.predict(line_X)                # ransac.predict调用前面拟合的线性模型，得出完美符合的点组成直线
plt.plot(line_X, line_y, color='red', label='RANSAC 拟合的直线')

# 绘制拟合的线性模型和真实线
plt.xlabel('X')
plt.ylabel('Y')
plt.title('RANSAC 线性拟合')
plt.legend()

# 显示图像
plt.show()
