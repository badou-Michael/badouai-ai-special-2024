import numpy as np
import matplotlib.pyplot as plt


# 归一化一：（0-1之间），公式'x_=(x−x_min)/(x_max−x_min)
def min_max_normalization(x):
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]


# 归一化二：（-1-1之间），公式x_=(x−x_mean)/(x_max−x_min)
def mean_normalization(x):
    return [(float(i) - np.mean(x)) / float(max(x) - min(x)) for i in x]


# 标准化:x∗=(x−μ)/σ
def z_score_normalization(x):
    x_mean = np.mean(x)
    s2 = sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x)
    return [(i - x_mean) / s2 for i in x]


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1 = []
# for i in l:
#     i+=2
#     l1.append(i)
# print(l1)
cs = []
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)
n = mean_normalization(l)
z = z_score_normalization(l)
print(n)
print(z)
'''
蓝线为原始数据，橙线为z
'''
plt.plot(l, cs)
plt.plot(z, cs)
plt.show()
