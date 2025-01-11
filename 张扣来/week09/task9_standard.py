import numpy as np
import matplotlib.pyplot as plt

# 归一化处理1
# 归一化（0~1）x_=(x−x_min)/(x_max−x_min)
# def Normalization1(x):
#     return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]
# 归一化处理
# 归一化（-1~1）x_=(x−x_mean)/(x_max−x_min)
def Normalization2(x):
    return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]
def z_score(x):
    #z-score标准化公式 x∗=(x−μ)/σ
    x_mean = np.mean(x)
    s2 = sum([(i-x_mean)**2 for i in x])/len(x)
    return [(i-x_mean)/s2 for i in x]
l = [-8,-9,-9,-7,-9,-6,-4,-5,9,3,3,3,3,4,7,7,7,7,6,6,6,6,8,8,8,8,3,3,2,3,4,5,6,6,6
    ,4,6,6,4,4,4,2,7,8,2,4,0,2,10,10,10,10,7,7,7,15,11,15,15,12]
cs= []
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)
n = Normalization2(l)
z = z_score(l)
print(n)
print(z)
plt.plot(l,cs)
plt.plot(z,cs)
plt.show()