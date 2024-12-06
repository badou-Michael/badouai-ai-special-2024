import numpy as np
import matplotlib.pyplot as plt

"""标准化两种方式"""
def Normalization1(x):
    '''归一化（0~1）'''
    '''x_=(x−x_min)/(x_max−x_min)'''
    return [(float(i)-min(x))/(max(x)-min(x)) for i in x]

def Normalization1(x):
    '''归一化（-1~1）'''
    '''x_=(x−x_mean)/(x_max−x_min)'''
    return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]

def z_score(x):
    """零均值归一化 经过处理后的数据均值为0，标准差为1（正态i分布）
       x∗=(x−μ)/σ
    """
    m= np.mean(x)
    s2=sum([(i-m)*(i-m) for i in x ])/len(x)
    return [(i-m)/s2 for i in x]


l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1=[]
#对应数值的个数
cs=[]
for i in l:
    c = l.count(i)
    cs.append(c)
print(cs)

n1 = Normalization1(l)
n2 = Normalization1(l)
z = z_score(l)
#绘图
plt.plot(l,cs)#蓝色 原图
plt.plot(n1,cs)#（0~1）
plt.plot(n2,cs)#（-1~1）
plt.plot(z,cs)#橙色 零均值归一化
plt.show()
