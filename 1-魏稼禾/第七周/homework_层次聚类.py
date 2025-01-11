# 层次聚类比较简单，类与类逐个计算距离，
# 距离最近的两个类合并成一个类，作为新类加入数据中，继续计算类与类的距离
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

X = [[1,2],[3,2],[4,4],[1,2],[1,3],[4,6],[23,59],[33,4],[22,57]]
Z = linkage(X,"ward")
print(Z)
f = fcluster(Z, 3, "maxclust")
print(f)
fig = plt.figure(figsize = (5,3))
dn = dendrogram(Z)
plt.show()