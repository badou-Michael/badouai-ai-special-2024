import numpy as np
import matplotlib.pyplot as plt

# 标准化函数（Z-score）
def z_score(x):
    '''z-score标准化：x* = (x - μ) / σ'''
    x_mean = np.mean(x)  # 计算均值
    x_std = np.std(x)    # 计算标准差
    return [(i - x_mean) / x_std for i in x]


# 数据集
l = [-10, 4,  5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 16, 30]

# 计算标准化后的数据
z = z_score(l)

# 打印标准化后的数据
print("Original Data:", l)
print("Standardized Data (z-score):", z)

# 计算每个值出现的频率
cs = []
for i in l:
    c = l.count(i)
    cs.append(c)

plt.plot(l, cs, label="Original Data")
plt.plot(z, cs, label="Standardized Data (z-score)", linestyle="--")
plt.legend()
plt.show()
