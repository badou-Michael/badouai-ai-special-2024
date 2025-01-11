import numpy as np
import matplotlib.pyplot as plt
import math


# d归一化方法一
def normalize(data):
    num = []
    for i in data:
        num.append((float(i) - np.mean(data)) / float(max(data) - min(data)))
    return num


# 归一化方法二
def normalize2(data):
    return [(float(i) - min(data)) / float(max(data) - min(data)) for i in data]


# 标准化方法
def standardization(data):
    ''' 另一种标准化方法： x∗=(x−μ)/σ'''
    # x_mean = np.mean(data)
    # s2 = sum([(i - np.mean(data)) * (i - np.mean(data)) for i in data]) / len(data)
    # s2 = math.sqrt(s2)
    # print('标准差：%s' % s2)
    # return [(i - x_mean) / s2 for i in data]

    '''便准化方法'''
    mean = np.mean(data)  # 求均值 μ
    std = np.std(data)  # 求标准差 σ
    print('标准差：%s' % std)
    # 公式：z=(x-μ)/σ
    z = [(float(z) - mean) / std for z in data]
    return z


l = [10, 20, 30, 40, 50]
count = 0
for i in l:
    count += i
print('总和:%s' % count)
norm_num = normalize(l)
norm_num2 = normalize2(l)
standard_num = standardization(l)
print(norm_num)
print(norm_num2)
print('Z-score:%s' % standard_num)
