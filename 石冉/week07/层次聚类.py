from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
from matplotlib import pyplot as plt
import random

#随机生成一个二维列表
X=[[random.randint(1,10),random.randint(1,10)] for _ in range(10)]
print(X)

#使用linkage函数进行层次聚类，‘ward’方法：最小化每次合并距离的方差增加量
Z=linkage(X,'ward')
print(Z)

#使用fcluster函数提取聚类，阈值设置为5
f=fcluster(Z,5,'distance')

#绘制树状图
dn=dendrogram(Z)
fig=plt.figure(figsize=(9,6))
plt.show()
