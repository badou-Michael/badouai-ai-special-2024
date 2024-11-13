import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
# X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]
X = np.array(X)
Z = linkage(X, method='ward', metric='euclidean')

# 层次信息
fig1 = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)

# 聚类结果
f = fcluster(Z, t=1.5, criterion='distance')
fig2 = plt.figure(figsize=(5, 3))
plt.scatter(X[:, 0], X[:, 1], c=f, cmap='prism')  # 二维数据
# plt.scatter(range(len(X)), X, c=f, cmap='prism')  # 一维数据

plt.show()
