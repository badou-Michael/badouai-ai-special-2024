from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]

# 使用linkage函数进行层次聚类，方法选择为ward，这是一种最小方差法，旨在最小化每个聚类内的方差。
Z = linkage(X, 'ward')
# 使用fcluster函数将层次聚类结果Z转换为平聚类结果，阈值设置为4，单位为距离。
f = fcluster(Z, 4, 'distance')
# 创建一个大小为 5x3 的图形。
fig = plt.figure(figsize=(5, 3))
# 使用dendrogram函数绘制树状图
dn = dendrogram(Z)
# 打印层次聚类的结果矩阵Z
print(Z)
plt.show()