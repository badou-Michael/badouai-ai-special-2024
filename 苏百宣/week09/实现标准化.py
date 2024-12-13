# 实现标准化 author：苏百宣

import numpy as np
import matplotlib.pyplot as plt

# 1.实现归一化
# （0~1）
def Normalization(X):
    return [(float(i)-min(X))/(max(X)-min(X)) for i in X]
# （-1~1）
def Normalization2(X):
    return [(float(i)-np.mean(X))/(max(X)-min(X)) for i in X]
# 标准化
def z_score(x):
    '''x∗=(x−μ)/σ'''
    x_mean=np.mean(x)
    s2=sum([(i-np.mean(x))*(i-np.mean(x)) for i in x])/len(x)
    return [(i-x_mean)/s2 for i in x]

new_list = [12, -6, 23, 9, 17, 18, 21, -1, 0, 26, 19, 8, 5, 30, 27, 22, 20, 15, 24, 13, -4, -10, 7, 14, -5, 2, 3, -2, 11, 6, 25, -7, 4, -8, 28, 10, 16, 29, -3, 1, -9, 7, 23, 3, 18, 13, -1, 8, 30]
cs = [new_list.count(i) for i in new_list]
n = Normalization2(new_list)  # 使用 [-1, 1] 归一化
z = z_score(new_list)  # 使用标准化
plt.plot(new_list, cs, label='Original Data')  # 蓝线：原始数据
plt.plot(z, cs, label='Standardized Data')    # 橙线：标准化数据
plt.legend()
plt.show()
