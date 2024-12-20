import numpy as np

# 数据
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

# 计算均值和标准差
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

# 进行z-score标准化
z_scored_data = (data - mean) / std

print("原始数据:", data)
print("标准化后的数据:", z_scored_data)
