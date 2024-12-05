import numpy as np
import matplotlib.pyplot as plt


# 归一化 (0-1)
def Normalization1(x):
#     x = (x- min) /  (max  -  min )
    cha = float(max(x)-min(x))
    return [(float(i)-min(x))/cha for i in x]

# 归一化 (-1 ~ 1)
def Normalization2(x):
#     x = (x- mean) /(max  -  min )
    return [(float(i-np.mean(x))/float(max(x)-min(x))) for i in x]
#
# 标准化:标准化是原始分数减去平均数然后除以标准差
def z_score(x):
#     x 符合 (x-u)/sigma
    x_mean = np.mean(x)
    # s2 标准差
    s2=sum([(i-np.mean(x))*(i-np.mean(x)) for i in x])/len(x)
    return [(i-x_mean)/s2 for i in x]


l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
m=Normalization1(l)
print(m)
n=Normalization2(l)
print(n)
z=z_score(l)
print(z)