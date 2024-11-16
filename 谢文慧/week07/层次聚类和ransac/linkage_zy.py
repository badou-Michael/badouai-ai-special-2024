from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster

# https://blog.csdn.net/weixin_46713695/article/details/125987933
X = [[1,2],[3,2],[4,4],[1,2],[1,3]]

# method = ‘single’最近邻点法
# method = 'complete' 最远邻点法
# method = 'ward' （沃德方差最小化算法）
# Z中返回的是层次聚类矩阵，第1个和第2个元素是每一步合并的两个簇
# 第3个元素是这些簇之间的距离，第4个元素是新簇的大小——包含的原始数据点的数量.Z
Z = linkage(X, 'single')
# print(Z)
# scipy.cluster.hierarchy.fcluster 可用于展平树状图，从而将原始数据点分配给单个簇
f = fcluster(Z,t=2,criterion='distance')
print(f)
fig = plt.figure(figsize=(5, 3))
# 画聚类树
dn = dendrogram(Z)
print(Z)
plt.show()