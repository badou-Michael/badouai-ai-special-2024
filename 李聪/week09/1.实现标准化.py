import numpy as np

# 定义一个标准化函数
def z_score(x):
    """
    z-score标准化公式：z = (x - μ) / σ
    x: 输入数据列表
    """
    x_mean = np.mean(x)  # 计算均值
    x_std = np.std(x)    # 计算标准差
    return [(i - x_mean) / x_std for i in x]  # 标准化公式

# 示例数据
data = [10, 20, 30, 40, 50]

# 调用标准化函数
standardized_data = z_score(data)

print("原始数据：", data)
print("标准化后数据：", standardized_data)
