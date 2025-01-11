# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def Normalization1(x):
    '''归一化（0~1）
    输入: x (list or array-like)
    输出: 归一化后的数据 (list)
    公式: x_ = (x - x_min) / (x_max - x_min)
    '''
    if not x:
        raise ValueError("输入数据不能为空")
    x_min = min(x)
    x_max = max(x)
    return [(float(i) - x_min) / float(x_max - x_min) for i in x]

def Normalization2(x):
    '''归一化（-1~1）
    输入: x (list or array-like)
    输出: 归一化后的数据 (list)
    公式: x_ = (x - x_mean) / (x_max - x_min)
    '''
    if not x:
        raise ValueError("输入数据不能为空")
    x_mean = np.mean(x)
    x_min = min(x)
    x_max = max(x)
    return [(float(i) - x_mean) / float(x_max - x_min) for i in x]

def z_score(x):
    '''标准化（Z-score）
    输入: x (list or array-like)
    输出: 标准化后的数据 (list)
    公式: x* = (x - μ) / σ
    '''
    if not x:
        raise ValueError("输入数据不能为空")
    x_mean = np.mean(x)
    s2 = np.std(x)
    return [(i - x_mean) / s2 for i in x]

# 准备数据
l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
cs = [l.count(i) for i in l]

# 调用预处理函数
normalized_data_01 = Normalization1(l)
normalized_data_neg11 = Normalization2(l)
standardized_data = z_score(l)

# 打印结果
print(normalized_data_neg11)
print(standardized_data)

# 绘制图表
try:
    plt.plot(normalized_data_01, cs, label='归一化（0~1）')
    plt.plot(normalized_data_neg11, cs, label='归一化（-1~1）')
    plt.plot(l, cs, label='原始数据')
    plt.plot(standardized_data, cs, label='标准化（Z-score）')
    plt.legend()
    plt.show()
except Exception as e:
    print(f"绘图时发生错误: {e}")
