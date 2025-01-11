#归一化与标准化

import numpy as np

# 最小值归一化,x=(x−x_min)/(x_max−x_min)
def guiyi_1(x):
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]
# 均值归一化,x=(x−x_mean)/(x_max−x_min)
def guiyi_2(x):
    x_mean = np.mean(x)
    return [(float(i) - x_mean) / float(max(x) - min(x)) for i in x]

# 标准化
def biaozhun(x):
    x_mean = np.mean(x)  # 计算均值
    s = np.std(x) # 计算标准差
    return [(i - x_mean) / s for i in x]

Y =[20,82,5,26,52,84,7,71,16,45,82,36,9,58,62,19,3,8,46,58,1,28,37,46,81]

x1=guiyi_1(Y)
x2=guiyi_2(Y)
x3=biaozhun(Y)

print('最小值归一化结果 (x1):')
print(x1)
print('均值归一化结果 (x2):')
print(x2)
print('标准化结果 (x3):')
print(x3)
