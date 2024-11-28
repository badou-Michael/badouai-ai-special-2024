import numpy as np
import matplotlib.pyplot as plt

# 归一化的两种方式
def Normalization1(x):
    '''归一化（0~1）'''
    '''x_=(x−x_min)/(x_max−x_min)'''
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]

def Normalization2(x):
    '''归一化（-1~1）'''
    '''x_=(x−x_mean)/(x_max−x_min)'''
    return [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]

# 标准化
def z_score(x):
    '''x*=(x−μ)/σ'''
    x_mean = np.mean(x)
    s2 = np.std(x)  # 使用numpy的std函数计算标准差，简化代码
    return [(i - x_mean) / s2 for i in x]

# 自定义数据
data = [12, 15, 12, 13, 14, 16, 16, 15, 14, 14, 16, 17, 18, 18, 16, 19, 19, 20, 19, 20, 21, 20, 22, 22, 21, 21, 19, 22, 25, 26, 23, 23, 24, 25, 24, 27, 28, 27, 30, 35]

# 计算频数
freq_count = []
for i in data:
    count = data.count(i)
    freq_count.append(count)

# 进行两种归一化
normalized1 = Normalization1(data)
normalized2 = Normalization2(data)

# 进行标准化
z_scaled = z_score(data)

# 打印结果
print("归一化（0~1）:", normalized1)
print("归一化（-1~1）:", normalized2)
print("标准化结果:", z_scaled)

# 可视化原始数据、标准化数据、归一化数据
plt.plot(data, freq_count, label="原始数据", color="blue")
plt.plot(z_scaled, freq_count, label="标准化数据", color="orange")
plt.plot(normalized2, freq_count, label="归一化（-1~1）", color="green")
plt.legend()
plt.title("数据归一化与标准化")
plt.xlabel("数据值")
plt.ylabel("频数")
plt.show()
