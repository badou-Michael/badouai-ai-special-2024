import numpy as np
import matplotlib.pyplot as plt
#归一化的两种方式
def Normalization1(x):
    '''归一化（0~1）'''
    '''x_=(x−x_min)/(x_max−x_min)'''
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]
def Normalization2(x):
    '''归一化（-1~1）'''
    '''x_=(x−x_mean)/(x_max−x_min)'''
    return [(float(i)-np.mean(x))/(max(x)-min(x)) for i in x]
#标准化
def z_score(x):
    '''x∗=(x−μ)/σ'''
    x_mean=np.mean(x)
    s2=sum([(i-np.mean(x))*(i-np.mean(x)) for i in x])/len(x)
    return [(i-x_mean)/s2 for i in x]
 
data = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10,
   10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

data_count=[]
for i in data:
    c = data.count(i)
    data_count.append(c)
print(f'统计每个数据出现的次数为{data_count}')
mean_nor = Normalization1(data)
min_nor = Normalization2(data)
z = z_score(data)
print(f'第一种归一化的结果：{mean_nor}')
print(f'第二种归一化的结果：{min_nor}')
print(f'第三种归一化的结果：{z}')
'''
蓝线为原始数据，橙线为z
'''
# plot(横坐标,纵坐标)
plt.plot(data,data_count)
plt.plot(mean_nor,data_count)
plt.plot(min_nor,data_count)
plt.show()
