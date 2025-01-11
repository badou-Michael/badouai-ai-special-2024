"""
yzc-层次聚类
1.linkage 生成链接矩阵，计算类间距离，记录层次信息
2.fcluster 生成聚类标签
    fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None)
    Z--链接矩阵
    t--阈值，根据criterion的值，如果 criterion='maxclust'，则 t 表示最大聚类数量。如果 criterion='distance'，则 t 表示最大距离阈值。
    criterion--决定阈值t的标准：'distance': 基于距离阈值。'maxclust': 基于最大聚类数量。
3.dendrogram 绘制层次聚类树
"""
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster

X = [[1,2],[3,2],[4,4],[1,2],[1,3],[10,10],[11,11],[12,12]]
Z = linkage(X,'ward')
f = fcluster(Z,5,'distance')
print(f)

plt.figure(figsize=(5,3))
dendrogram(Z)
print(Z)
plt.show()


