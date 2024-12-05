import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

data = np.random.randint(0, 60, size=(70, 2))
# 进行层次聚类
Hier = linkage(data, method="single")
print(Hier)
# 绘制层次聚类的树状图
plt.subplot(1, 2, 1)
dendrogram(Hier)
lable = fcluster(Hier, t = 3, criterion="maxclust")
print(lable)
plt.subplot(1, 2, 2)
color = ['r', 'g', 'b']
for i in range(3):
    row_indices = np.where((lable-1) == i)
    print(row_indices)
    cluster = data[row_indices]
    print(cluster)
    plt.scatter(cluster[:, 0], cluster[:, 1], color= color[i])
plt.show()
