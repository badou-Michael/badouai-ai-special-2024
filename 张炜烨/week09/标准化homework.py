import numpy as np
import matplotlib.pyplot as plt

# 数据归一化方法
def normalize(x, mode="0-1"):
    """
    数据归一化
    Parameters:
        x: 输入数据列表或数组
        mode: 归一化模式 ("0-1" 或 "-1~1")
    Returns:
        归一化后的数据
    """
    x_min, x_max = min(x), max(x)
    if mode == "0-1":
        return [(i - x_min) / (x_max - x_min) for i in x]
    elif mode == "-1~1":
        x_mean = np.mean(x)
        return [(i - x_mean) / (x_max - x_min) for i in x]
    else:
        raise ValueError("Invalid mode. Use '0-1' or '-1~1'.")

# 数据标准化方法
def z_score(x):
    """
    数据标准化 (Z-score)
    Parameters:
        x: 输入数据列表或数组
    Returns:
        标准化后的数据
    """
    x_mean = np.mean(x)
    x_std = np.std(x)
    return [(i - x_mean) / x_std for i in x]

# 数据统计
def count_occurrences(x):
    """
    统计每个值的出现次数
    Parameters:
        x: 输入数据列表或数组
    Returns:
        出现次数列表，与输入数据一一对应
    """
    return [x.count(i) for i in x]

if __name__ == "__main__":
    # 原始数据
    l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 
         10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 
         12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

    # 统计数据出现次数
    cs = count_occurrences(l)

    # 数据归一化和标准化
    normalized_data = normalize(l, mode="-1~1")  # 使用 -1~1 归一化
    z_scored_data = z_score(l)                  # 标准化

    # 打印结果
    print("归一化数据 (-1~1):", normalized_data)
    print("标准化数据 (Z-score):", z_scored_data)

    # 绘制数据
    plt.figure(figsize=(10, 6))
    plt.plot(l, cs, label="Original Data (Counts)", color="blue", marker="o")
    plt.plot(z_scored_data, cs, label="Z-score Standardization", color="orange", marker="x")
    plt.xlabel("Data")
    plt.ylabel("Counts")
    plt.title("Data Normalization and Standardization")
    plt.legend()
    plt.grid()
    plt.show()