# 导包
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

"""
linkage(y, method='single', metric='euclidean')共包含3个参数
1.y是距离矩阵，可以是1为压缩向量（距离向量），也可以是2为观测向量（坐标矩阵）
  若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2．method是指计算类间距离的方法。

fcluster(Z, t, criterion='inconsistent',depth=2, R=None, monocrit=None)
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息;
2.t是一个聚类的阈值"The threshold to apply when forming flat clusters"。

"""
X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
# 用于计算类间距离，生成层次聚类树状图的基础数据。
Z = linkage(X,'ward')
# 用于根据给定的阈值将层次聚类树状图划分为不同的簇。
f = fcluster(Z,4,'distance')
'''
4是聚类的阈值，表示距离小于4的样本点会被归为一类。
distance是聚类的标准，表示使用距离作为划分标准。
'''
fig = plt.figure(figsize=(5,3))
# 绘制层次聚类树状图。
dn = dendrogram(Z)
print(Z)
plt.show()