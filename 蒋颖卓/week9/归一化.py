import numpy as np
import matplotlib.pyplot as plt


# 归一化两种方式

def Normalization1(x):
    # 归一化(0-1)
    return [(float(i) - min(x)) / float(max(x) - min(x)) for i in x]


def Normalization0(x):
    """归一化（-1-1）"""
    return [(float(i) - np.mean(x)) / float(max(x) - min(x)) for i in x]


def z_score(x):
    """"标准化"""
    x_mean = np.mean(x)
    s2 = sum((i - x_mean) ** 2 for i in x) / len(x)
    sigma = np.sqrt(s2)
    return [(i - x_mean) / sigma for i in x]


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1 = []
cs = []

n = Normalization0(l)  # 归一化
z = z_score(l)  # 标准化
print(n)
print(z)

for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)
plt.plot(l, cs)
plt.plot(n, cs)
plt.show()
