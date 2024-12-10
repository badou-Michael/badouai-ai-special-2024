from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster  #层次聚类专用包

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。

'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''
X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
#linkage用来读取数据，ward是指一个计算类间距离的方法
Z = linkage(X,"ward")
#真正计算聚类的函数,4为限定的循环次数
f =fcluster(Z,4,"distance")
#可视化部分
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()

