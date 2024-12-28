import numpy as np
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
import matplotlib.pyplot as plt

# 示例数据
data = np.array([[1,2],[2,3],[3,4],[5,8],[8,8]])

#层次聚类
Z = linkage(data,method='ward',metric='euclidean')

#提取簇划分结果，距离阈值为4
f = fcluster(Z, 4, 'distance')

#输出簇的标签
print(f)#每个数据点的簇标签

#绘制树状图
plt.figure(figsize=(8,6))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
plt.show()
