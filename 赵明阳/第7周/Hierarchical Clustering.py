import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
# 凝聚式层次聚类
# 首先，使用numpy生成一些随机的二维数据点,生成一些示例数据,
np.random.seed(0)
data = np.random.randn(10, 2)

# 然后，使用scipy.cluster.hierarchy模块中的linkage函数进行层次聚类，这里使用了'ward'方法，它最小化方差，使得合并后的聚类内部方差增加最小。计算距离矩阵并进行层次聚类
Z = linkage(data, method='ward')

# 绘制树形图,以便直观地观察聚类的层次结构。
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data points')
plt.ylabel('Distance')
plt.show()