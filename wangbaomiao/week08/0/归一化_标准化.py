# -*- coding: utf-8 -*-
# time: 2024/11/6 17:20
# file: 归一化_标准化.py
# author: flame
import numpy as np
import matplotlib.pyplot as plt

""" 
定义三个函数：Normailzation1、Normailzation2 和 z_score，用于对数据进行归一化和标准化处理。
首先创建一个包含一系列整数的列表l，然后计算每个元素出现的次数并存储在lc列表中。
接着使用这三个函数分别对列表l进行处理，并打印处理后的结果。
最后使用matplotlib库绘制原始数据及处理后的数据图形，展示不同处理方法的效果。
"""

""" 定义函数Normalization1，用于数据的归一化处理。 """
def Normailzation1(x):
    """ 返回一个新的列表，其中每个元素是原列表中的元素减去最小值后除以最大值与最小值的差。 """
    return [(float(i)-min(x)/float(max(x)-min(x))) for i in x]

""" 定义函数Normalization2，用于数据的另一种归一化处理。 """
def Normailzation2(x):
    """ 返回一个新的列表，其中每个元素是原列表中的元素减去平均值后除以最大值与最小值的差。 """
    return [(float(i) - np.mean(x)/(max(x)-min(x))) for i in x]

""" 定义函数z_score，用于计算列表中每个元素的z-score标准化值。 """
def z_score(x):
    """ 计算列表元素的平均值。 """
    x_mean = np.mean(x)
    """ 计算列表元素的方差。 """
    s2 = sum([(i-np.mean(x))*(i-np.mean(x)) for i in x])/len(x)
    """ 返回一个新的列表，其中每个元素是原列表中的元素减去平均值后除以方差。 """
    return [(i-np.mean(x))/s2 for i in x]

""" 创建一个包含一系列整数的列表l，用于后续的归一化和标准化处理。 """
l=[-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

""" 创建一个空列表lc，用于存储列表l中每个元素出现的次数。 """
lc = []

""" 遍历列表l中的每个元素，计算其出现次数，并将结果添加到lc列表中。 """
for i in l:
    """ 计算当前元素i在列表l中出现的次数。 """
    c = l.count(i)
    """ 将当前元素i的出现次数添加到lc列表中。 """
    lc.append(c)

""" 使用Normalization1函数对列表l进行归一化处理。 """
n = Normailzation1(l)
""" 使用Normalization2函数对列表l进行另一种归一化处理。 """
n2 = Normailzation2(l)
""" 使用z_score函数对列表l进行z-score标准化处理。 """
z = z_score(l)

""" 打印归一化和标准化后的结果。 """
print("n : ", n)
print("n2 : ", n2)
print("z : ", z)

""" 使用matplotlib库绘制原始数据及处理后的数据图形。 """
""" 绘制原始数据l与lc的关系图，并设置颜色和标签。 """
plt.plot(l, lc, color='blue', label='l')
""" 绘制归一化数据n与lc的关系图，并设置颜色和标签。 """
plt.plot(n, lc, color='red', label='N1')
""" 绘制另一种归一化数据n2与lc的关系图，并设置颜色和标签。 """
plt.plot(n2, lc, color='green', label='N2')
""" 绘制标准化数据z与lc的关系图，并设置颜色和标签。 """
plt.plot(z, lc, color='purple', label='Z-Score')

""" 添加图例，设置图例位置为左上角。 """
plt.legend(loc='upper left')

""" 显示绘制的图形。 """
plt.show()
