import numpy as np
from matplotlib import pyplot as plt


def normalization1(x):
    """归一化(0-1)"""
    """x_=(x−x_min)/(x_max−x_min)"""
    return [(i - min(x))/max(x) - min(x) for i in x]


def normalization2(x):
    """归一化(-1-1)"""
    """x_=(x-x_mean)/(x_max-x_min)"""
    return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]
    # return [i - np.mean(x)/max(x) - min(x) for i in x]


# 标准化：用于将数据转换为均值为 0、标准差为 1 的标准正态分布
def z_score(x):
    """z-score:x∗=(x−μ)/σ"""
    """x_=(x-x_mean)/(x_std)"""
    # x_mean=np.mean(x)
    # print("mean:", x_mean)
    # s2=sum([(i-np.mean(x))*(i-np.mean(x)) for i in x])/len(x)
    # print("s2:", s2)
    # return [(i-x_mean)/s2 for i in x]
    # 平均值
    x_mean = np.mean(x)
    print("mean:", x_mean)
    # 标准差
    s2 = sum([(i-x_mean)**2 for i in x])/len(x)
    print("s2:", s2)
    # 标准化
    return [(i - x_mean)/s2 for i in x]

l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
# l1 = []
# for i in l:
#     i+=2
#     l1.append(i)
# print(l1)
cs=[]
for i in l:
    c=l.count(i)
    cs.append(c)
print("cs:", cs)
print("mean:", np.mean(l))
n = normalization2(l)
print("normalization2:", n)
z = z_score(l)
print("z_score:", z)
plt.plot(l, cs)  # l:x轴, cs:y轴
plt.plot(z, cs)  # z:x轴, cs:y轴
plt.show()
