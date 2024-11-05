import numpy as np
import matplotlib.pyplot as plt
import math

# 归一化
def minmaxNorm(x):
    return [(xi-min(x))/(max(x)-min(x)) for xi in x]

def meanNorm(x):
    return [(xi-np.mean(x))/(max(x)-min(x)) for xi in x]

# 标准化
def zScoreNorm(x):
    mean = np.mean(x)
    sigma = math.sqrt(sum([pow(xi-mean, 2) for xi in x])/len(x))
    return [(xi-mean)/sigma for xi in x]

# test
l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1=[]
# for i in l:
#     i+=2
#     l1.append(i)
# print(l1)
cs=[]
for i in l:
    c=l.count(i)
    cs.append(c)
print(cs)

n=meanNorm(l)
z=zScoreNorm(l)
print(n)
print(z)
'''
蓝线为原始数据，橙线为z
'''
plt.plot(l,cs)
plt.plot(z,cs)
plt.show()
