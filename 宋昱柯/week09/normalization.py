import numpy as np
import matplotlib.pyplot as plt

def normalization1(x):
    """归一化(0~1)"""
    return (x - min(x)) / (max(x) - min(x))
    # return np.asarray([(i - min(x)) / (max(x) - min(x)) for i in x])


def normalization2(x):
    """归一化(-1~1)"""
    return 2 * (x - np.mean(x)) / (max(x) - min(x))
    # return np.asarray(2 * [(i - np.mean(x)) / (max(x) - min(x)) for i in x])


def z_score(x):
    """标准归一化"""
    x_mean = np.mean(x)
    std = ((np.sum([(i - x_mean) ** 2 for i in x])) / (len(x))) ** (1 / 2)
    return np.asarray([(i - x_mean) / std for i in x])


l = [
    -10,
    5,
    5,
    6,
    6,
    6,
    7,
    7,
    7,
    7,
    8,
    8,
    8,
    8,
    8,
    9,
    9,
    9,
    9,
    9,
    9,
    10,
    10,
    10,
    10,
    10,
    10,
    10,
    11,
    11,
    11,
    11,
    11,
    11,
    12,
    12,
    12,
    12,
    12,
    13,
    13,
    13,
    13,
    14,
    14,
    14,
    15,
    15,
    30,
]
cs = []

for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)

n = normalization2(l)
z = z_score(l)
print(n)
print(z)

plt.plot(l, cs)
plt.plot(z, cs)
plt.show()
