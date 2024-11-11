'''
WEEK7作业：
1.实现层次聚类 2.实现ransac
'''

from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
import cv2
import numpy as np

#1.实现层次聚类
#数据 X 中包含 5 个点，每个点具有两个维度。
X = [[10,20],[13,12],[14,14],[12,22],[11,13]]
#使用 ward 方法计算聚类距离，并生成链式矩阵 Z，每一行记录了两个聚类间合并的相关信息。链式矩阵 Z 的输出是聚类合并信息的矩阵。
Z = linkage(X, 'ward')
#fcluster 用于生成扁平聚类结果，其中 4 表示聚类的距离阈值，distance 表示用距离作为判断依据。f 的结果是每个点对应的簇标签。
f = fcluster(Z,2,'distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()

#2.实现ransac
# 假设有一些配对的关键点 src_pts 和 dst_pts
src_pts = np.float32([[10, 20], [15, 25], [40, 50], [100, 150]])
dst_pts = np.float32([[12, 22], [18, 28], [43, 53], [102, 152]])

# 使用 RANSAC 估计变换矩阵
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
print("Homography matrix with RANSAC:\n", M)
