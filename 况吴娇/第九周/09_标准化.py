import numpy as np
import matplotlib.pyplot as plt
###归一化方式
def Normalization1(x):
    '''归一化（0~1）'''
    '''x_=(x−x_min)/(x_max−x_min)'''
    return [ (float(i) - min(x))/(max(x)- min(x)) for i in x]
def Normalization2(x):
    '''归一化（-1~1）'''
    '''x_=(x−x_mean)/(x_max−x_min)'''
    return [(float(i) - np.mean(x)) / (max(x) - min(x)) for i in x]
#标准化
def z_score(x):
    '''x∗=(x−μ)/σ'''
    #z-score标准化（零均值归一化zero-mean normalization）： 中μ是样本的均值， σ是样本的标准差
    # 计算数据x的平均值x_mean。
    # 计算数据x的方差s2，方差是每个数值减去平均值的平方和除以数值的个数。
    # 对于数据中的每个数值i，用i减去平均值，然后除以方差的平方根（即标准差），得到标准化后的数值。
    x_mean=np.mean(x)
    s2 = sum([(i - x_mean) ** 2 for i in x]) / len(x)
    #s2 =sum ( [ (i - x_mean) *(i - x_mean)for i in x ])/len(x)
    return [(i-x_mean)/s2 for i in x]
l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1=[]
# for i in l:
#     i+=2
#     l1.append(i)
# print(l1)
cs=[]
for i in l:
    c=l.count(i) #对于列表l中的每个数值i，计算它在l中出现的次数c。
    cs.append(c) #将次数c添加到列表cs中。
print(cs)
n=Normalization2(l)
z=z_score(l)
print(n)
print(z)
plt.plot(l,cs)
plt.plot(z,cs)
# 第一个图表是原始数据l和每个数值出现的次数cs。
# 第二个图表是标准化后的数据z和每个数值出现的次数cs。
plt.show()  #：显示图表。

'''
蓝线为原始数据，橙线为z
matplotlib 会使用默认的颜色序列来绘制线条。
蓝色 ('b')
橙色 ('orange')
绿色 ('g')
红色 ('r')
紫色 ('purple')
棕色 ('brown')
粉红色 ('pink')
灰色 ('gray')
橄榄色 ('olive')
青色 ('cyan')
等等，直到用完为止，然后会循环使用。
plt.plot(l, cs, 'b') 绘制原始数据 l 和对应的计数 cs，并使用蓝色（'b'）线条。
plt.plot(z, cs, 'orange') 绘制标准化后的数据 z 和对应的计数 cs，并使用橙色（'orange'）线条。
'''
