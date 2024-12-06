import numpy as np
import matplotlib.pyplot as plt
#归一化的两种方式

'''
 归一化（0~1）
 x=(x−x_min)/(x_max−x_min)
'''
def Normalization1(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]
'''
 零均值归一化（0~1）
 x=(x−x_mean)/(x_max−x_min)
'''
def Normalization2(x):
    return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]


#标准化
    '''
    x=(x−μ)/σ
    '''
def z_score(x):
    #数据集的均值 x_mean=np.mean(x)
    x_mean = sum(x)/len(x)

    #方差s2
    s2 = sum([(i-x_mean)*(i-x_mean) for i in x])/len(x)

    #标准差
    #s = np.sqrt(s2)

    z = [(i-x_mean)/s2 for i in x]
    return z


l = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
cs = []
print(type(l))
for i in l:
    # 统计某个元素在列表中出现的次数
    c = l.count(i)
    cs.append(c)
print(cs)
n = Normalization2(l)
z = z_score(l)
print(n)
print(z)
'''
蓝线为原始数据，橙线为z
'''
plt.plot(l, cs)
plt.plot(z, cs)
plt.show()
