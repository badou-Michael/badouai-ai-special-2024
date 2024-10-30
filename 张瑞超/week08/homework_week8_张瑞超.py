import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def normalization1(x):
    x_min, x_max = min(x), max(x)
    return [(i - x_min) / (x_max - x_min) for i in x]


def normalization2(x):
    return [(i - np.mean(x)) / (max(x) - min(x)) for i in x]


def z_score(x):
    x_mean = np.mean(x)
    s2 = sum([(i - np.mean(x)) * (i - np.mean(x)) for i in x]) / len(x)
    return [(i - x_mean) / s2 for i in x]


# 原始数据
l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9,
     10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12,
     12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

hash = Counter(l)
cs = [hash[i] for i in l]

print(cs)
n = normalization2(l)
z = z_score(l)
print(n)
print(z)

plt.plot(l, cs)
plt.plot(z, cs)
plt.show()

