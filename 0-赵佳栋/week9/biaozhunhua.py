#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：biaozhunhua.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/11/22 12:15
'''

import numpy as np
import matplotlib.pyplot as plt

# 归一化
def Normalization1(x):
    m = [(float(i) - min(x)) / (max(x) - min(x))for i in x]
    return m

def Normalization2(x):
    m = [(float(i) - np.mean(x)) / (max(x) - min(x))for i in x]
    return m

# 零均值归一化
def z_score(x):
    x_mean = np.mean(x)
    s2 = np.sqrt(sum([(float(i)-x_mean) ** 2for i in x]) / len(x))
    m = [(i - x_mean) / s2 for i in x]
    return m

l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10,
     11, 11, 11, 11, 11, 11,12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
cs = []
for i in l:
    # 对每个元素使用count方法计算其出现的次数，并将这些次数存储在列表cs中
    c = l.count(i)
    cs.append(c)
print(cs)
n = Normalization2(l)
z = z_score(l)
print(n)
print(z)

plt.plot(l, cs)
plt.plot(z, cs)
plt.show()