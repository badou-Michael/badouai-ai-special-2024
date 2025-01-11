import numpy as np

#三种方式实现标准化

#最小-最大归一化到[0,1]区间
def min_max_normalization(data):
    # 计算最小值和最大值
    min_val = np.min(data)
    max_val = np.max(data)

    # 执行最小-最大归一化
    normalized_data = (data - min_val) / (max_val - min_val)

    return normalized_data

#均值-最小-最大归一化到[-1,1]区间
def normalize_data(data):
    # 计算最小值和最大值
    min_val = min(data)
    max_val = max(data)

    # 计算均值
    mean_val = sum(data) / len(data)

    # 应用归一化公式
    normalized_data = [(x - mean_val) / (max_val - min_val) for x in data]

    return normalized_data



#Z-score 标准化到均值0和方差1附近
def z_score_normalization(data):
    # 计算均值
    mean = sum(data) / len(data)

    # 计算方差
    variance = sum((x - mean) ** 2 for x in data) / len(data)

    # 计算标准差
    std_dev = variance ** 0.5

    # 计算Z-Score
    z_scores = [(x - mean) / std_dev for x in data]

    return z_scores, mean, std_dev


# 示例数据
data = [10, 20, 30, 40, 50]

# 调用函数进行最小-最大归一化
min_max_normalized_data = min_max_normalization(data)

# 调用函数进行归一化
normalized_data = normalize_data(data)

# 调用函数进行 Z-Score 标准化
z_scores, mean, std_dev = z_score_normalization(data)



# 打印标准化后的数据
print("Min-Max Normalized Data:", min_max_normalized_data)
print("Normalized Data:", normalized_data)
print("Z-Scores:", z_scores)
