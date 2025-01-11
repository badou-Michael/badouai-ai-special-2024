import numpy as np
from numpy import random
from sklearn.decomposition import PCA

if __name__ == "__main__":
    # 创建一个示例矩阵
    X = np.array([[1, 2, 3, 4],
                  [4, 5, 2, 2],
                  [7, 5, 9, 5],
                  [10, 15, 2, 29],
                  [15, 46, 12, 13],
                  [23, 21, 32, 30],
                  [11, 9, 43, 35],
                  [42, 45, 55, 11],
                  [9, 48, 12, 5],
                  [11, 21, 23, 14],
                  [8, 5, 45, 15],
                  [11, 12, 1, 21],
                  [21, 20, 12, 25]])

# 初始化PCA对象，设置要保留的主成分数量
pca = PCA(n_components=2)

# 对数据进行拟合和转换
X_reduced = pca.fit_transform(X)
print("降维后的矩阵：")
print(X_reduced)
