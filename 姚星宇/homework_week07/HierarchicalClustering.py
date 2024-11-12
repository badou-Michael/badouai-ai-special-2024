from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

# 数据集
X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]

# 进行层次聚类
Z = linkage(X, 'ward')  # 使用Ward方法计算类间距离

# 根据距离阈值形成平坦聚类
f = fcluster(Z, 4, criterion='distance')

# 打印层次聚类矩阵
print("层次聚类矩阵 Z:")
print(Z)

# 绘制层次树（Dendrogram）
fig = plt.figure(figsize=(8, 4))
dn = dendrogram(Z)
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# 绘制聚类结果
plt.figure(figsize=(8, 4))
plt.scatter([x[0] for x in X], [x[1] for x in X], c=f, cmap='viridis', s=50)
plt.title('Agglomerative Clustering Result')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()