import numpy as np
import matplotlib.pyplot as plt
# 归一化 0~1  x_=(x−x_min)/(x_max−x_min)
def Normalization1(x):
    return [(float(i) - min(x))/float(max(x)-min(x)) for i in x]
# 归一化 -1~1  x_=(x−x_mean)/(x_max−x_min)
def Normalization2(x):
    return [(float(i) - np.mean(x))/float(max(x)-min(x)) for i in x]
# 标准化
def z_score(x):
    x_mean = np.mean(x)
    s2=sum([(i-np.mean(x))*(i-np.mean(x)) for i in x])/len(x)
    return [(i-x_mean)/s2 for i in x]

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
cs=[]
for i in l:
    c = l.count(i)
    # print(c)
    cs.append(c)
# print(cs)
n1=Normalization1(l)
n2=Normalization2(l)
z=z_score(l)
print('归一化 0~1 :',n1)
print('归一化 -1~1： ',n2)
print('标准化：',z)
plt.plot(l,cs)
plt.plot(n1,cs)
plt.plot(n2,cs)
plt.plot(z,cs)
plt.show()
