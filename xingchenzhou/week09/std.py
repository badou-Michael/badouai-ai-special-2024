import matplotlib.pyplot as plt
# 示例数据
data = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12,
        12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

# 归一化
# 0-1
min_val = min(data)
max_val = max(data)
normalized_data = [(x - min_val) / (max_val - min_val) for x in data]

#标准化
mean = sum(normalized_data) / len(normalized_data)
variance = sum([(x - mean) ** 2 for x in normalized_data]) / len(normalized_data)
std_dev = variance ** 0.5
standardized_data = [(x - mean) / std_dev for x in normalized_data]

# 输出结果
print("原始数据:", data)
print("归一化数据:", normalized_data)
print("标准化数据:", standardized_data)

# 绘制原始数据、归一化数据、标准化数据的分布
plt.figure(figsize=(10, 6))
plt.plot(data, label="Original Data", marker='o')
plt.plot(normalized_data, label="Normalized Data (0-1)", marker='x')
plt.plot(standardized_data, label="Standardized Data (Z-score)", marker='s')
plt.legend()
plt.title("Data Transformation: Original, Normalized, Standardized")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid()
plt.show()

