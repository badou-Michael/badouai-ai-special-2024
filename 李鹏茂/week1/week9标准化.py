import numpy as np
import matplotlib.pyplot as plt

# 归一化 (-1~1)
def Normalization3(x):
    '''归一化（-1~1）'''
    '''x' = 2 * (x - x_min) / (x_max - x_min) - 1'''
    x_min = min(x)
    x_max = max(x)
    return [2 * (i - x_min) / (x_max - x_min) - 1 for i in x]

# 标准化
def z_score(x):
    '''标准化 (z = (x - μ) / σ)'''
    x_mean = np.mean(x)
    s2 = np.std(x)  # 使用标准差计算
    return [(i - x_mean) / s2 for i in x]

l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
cs = []
for i in l:
    c = l.count(i)
    cs.append(c)

# 归一化到 [-1, 1]
n3 = Normalization3(l)

# 标准化
z = z_score(l)

# 打印结果
print("归一化到 [-1, 1]：", n3)
print("标准化结果：", z)

# 绘图：蓝线为原始数据，橙线为标准化数据，绿线为归一化到 [-1, 1] 的数据
plt.plot(l, cs, label="原始数据", color='blue')
plt.plot(z, cs, label="标准化数据 (z-score)", color='orange')
plt.plot(n3, cs, label="归一化到 [-1, 1]", color='green')
plt.legend()
plt.show()
