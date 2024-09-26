import numpy as np

# 原始矩阵
X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
# 降维度
K = X.shape[1]-1
# 去中心化,相当于[x1,x2,...,xn] 每一列求平均，所以用0轴
xmean = X-X.mean(axis=0)
print("去中心化矩阵\n",xmean)
# 求协方差矩阵
cov_matrix = np.dot(xmean.T,xmean)/(X.shape[0]-1)
# 获取特征值与特征向量
eigen_value,eigen_vector = np.linalg.eig(cov_matrix)
# 特征值从大到小排序，取相对应索引的特征向量,这里需要根据降到的维度取
sort = np.argsort(-1*eigen_value)
print(sort)
reduce_eigen_vector = eigen_vector[:,sort[:K]]
pca_matrix = np.dot(xmean,reduce_eigen_vector)
print("降维后的矩阵\n",pca_matrix)