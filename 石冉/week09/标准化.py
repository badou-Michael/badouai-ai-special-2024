import numpy as np
import random
import matplotlib.pyplot as plt
import math

#随机生成一个包含100个数字的数组
X = []
while len(X) < 100:
    i = random.randint(0, 100)
    X.append(i)  # 这里不需要重新赋值给X
print('原始数据是：',X)

#对数据进行归一化
Y=[]
j=0
for i in range(0,100):
    j=(float(X[i])-min(X))/(max(X)-min(X))
    Y.append(j)
    i+=1

#对数据进行标准化,x-u/σ
y_mean=np.mean(Y)
s=0
for i in Y:
    s1=(i-y_mean)**2/len(Y)
    s=s+s1
sigma=math.sqrt(s)
for i in range(0,100):
    Y[i]=(Y[i]-y_mean)/sigma

print('标准化后的结果是：',Y)

#画图：红色是原数据，蓝色是标准化后的数据
plt.plot(X,'r')
plt.plot(Y,'b')
plt.show()
