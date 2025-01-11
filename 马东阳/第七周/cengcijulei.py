
# 层次聚类：
# 1:开始将每个样本视为一个簇；
# 2：每次按照一定的准则将距离最近的两个簇合并成一个新的簇
# 3：如此往复，直到所有的样本都属于一个簇。

# 方式 直接通过接口实现
# 导入包
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

# X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
data = [[16.9,0],[38.5,0],[39.5,0],[80.8,0],[82,0],[834.6,0],[116.1,0]]
Z = linkage(data, 'ward')
f = fcluster(Z,4,'distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()
