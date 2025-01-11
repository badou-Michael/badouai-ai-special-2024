import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# 数据
X = np.array([[1,2], [3,2], [4,4], [1,2], [1,3]])

# 层次聚类
Z = linkage(X, method='ward')

# 树状图
plt.figure(figsize=(8, 4))
dendrogram(Z, labels=np.arange(1, len(X)+1))
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

# 按距离划分聚类
clusters = fcluster(Z, t=3, criterion='distance')
print("Cluster labels:", clusters)
