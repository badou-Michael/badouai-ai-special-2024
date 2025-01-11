#导入相应的包

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import  pyplot as plt

#有一组二维点
X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
#计算连接矩阵
Z = linkage(X, 'ward')
print(Z)
f = fcluster(Z,4, 'distance')
fig = plt.figure(figsize=(5,3))
dn = dendrogram(Z)
plt.show()
