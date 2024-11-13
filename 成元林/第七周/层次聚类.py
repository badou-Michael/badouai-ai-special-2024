from scipy.cluster.hierarchy import linkage,dendrogram,fcluster,ward
import numpy as np
import matplotlib.pyplot as plt
#数据坐标点
X = np.array([[1, 2], [1, 4], [1, 0],
                    [1, 2], [2, 2], [2, 4],
                    [3, 2], [3, 3], [3, 5]])
"""
计算聚类数据的链接矩阵，method:
"single"：最简单的方法，计算两组数据点之间的单个距离
"complete"：计算两组数据点之间的最大距离。
"average"：计算两组数据点之间的平均距离。
"weighted"：类似于average方法，但是给予数据点更高的权重。
"centroid"：计算组中心之间的距离。
"ward"：使用各种统计量来衡量组间的差异，这是一种更复杂的方法，常用于散点图数据。
"""
Z = linkage(X, method='ward')
#使用fcluster函数进行聚类，阈值为2
f = fcluster(Z,2,criterion="distance")
print("f\n",f)
# 画图
dn = dendrogram(Z)
print("Z\n",Z)
plt.show()


