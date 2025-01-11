import numpy as np

def z_score_standardization(data):
    """
    实现标准化（Z-score Normalization）
    :param data: 输入数据列表或数组
    :return: 标准化后的数据列表
    """
    mean = np.mean(data)  # 计算均值
    std = np.std(data)    # 计算标准差
    return [(x - mean) / std for x in data]  # 按公式标准化

# 示例数据
data = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 
        10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 
        12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

# 标准化
standardized_data = z_score_standardization(data)

# 输出原始数据和标准化后的数据
print("原始数据:", data)
print("标准化数据:", standardized_data)
