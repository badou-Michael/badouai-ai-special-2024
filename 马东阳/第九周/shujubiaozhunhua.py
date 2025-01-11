import numpy as np
import matplotlib.pyplot as plt

# 归一化
# 0-1归一化
def MinMax(data):
    min_val = np.min(data)
    max_val = np.max(data)
    # 归于【0，1】之间
    norm_data = (data - min_val) / (max_val - min_val)
    return norm_data

# -1~1归一化
def MinMax2(data):
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    # 归于【-1，1】
    norm_data = (data - mean_val) / (max_val - min_val)
    return norm_data

# Z-Score标准化
def Z_Score(data):
    mean = np.mean(data)
    std_data = np.std(data)
    norm_data = (data - mean) / std_data
    return norm_data

# 数据测试
data_1 = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

cs = []
for i in data_1:
    c=data_1.count(i)
    cs.append(c)
print(cs)
n = MinMax2(data_1)
z = Z_Score(data_1)
print(n)
print(z)

# 蓝线为原始数据，橙线为z

plt.plot(data_1, cs)
plt.plot(z, cs)
plt.show()
