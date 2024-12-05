# 把输入数据边转化到同一个范围内，防止有些数据过大，计算梯度时梯度爆炸，
# 错过最小值，损失函数反复波动，不能收敛到最小值
import numpy as np
import matplotlib.pyplot as plt
#归一化的两种方法
def Normalization1(x):
    '''归一化范围（0,1）
    公式(xi-x_min)/(x_max-x_min)'''
    return [(float(i)-min(x))/(max(x)-min(x)) for i in x]

def Normalization2(x):
    '''归一化范围（-1,1）
    公式(xi-x_mean)/(x_max-x_min)'''
    return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]

# 标准化
# 范围（-1,1）， 平均为0，标准差为1  （x-x_mean）/方差
def z_score(x):
    x_mean =np.mean(x)
    s2 =sum([(i-x_mean)*(i-x_mean) for i in x])/len(x)
    return [(i-x_mean)/s2 for i in x]
l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
l1 = []
count_i= []
for i in l:
    c = l.count(i)
    count_i.append(c)
n = Normalization1(l)
z =z_score(l)
print(n)
print(z)
plt.plot(l, count_i)
plt.plot(z,count_i)
plt.show()
