'''
scipy.cluster.hierarchy.linkage
函数是 SciPy 库中用于层次聚类（Hierarchical Clustering）的一个关键函数。
scipy.cluster.hierarchy.
    linkage(y, method='single', metric='euclidean', optimal_ordering=False)
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。
(如何计算两个簇之间的距离 ，最短距离 single，最长距离complete，中间距离，簇平均法average)

'''
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

#数据集
X = [[0, 0], [0, 1], [1, 0], [0, 4], [0, 3], [1, 4], [4, 0], [3, 0], [4, 1], [4, 4], [3, 4], [4, 3]]
#调用层次聚类接口
c_z = linkage(X, method='ward')
print(c_z)
f = fcluster(c_z,1,criterion='distance')

fig = plt.figure(figsize=(5, 3))
dn = dendrogram(c_z)
print(c_z)

plt.show()


