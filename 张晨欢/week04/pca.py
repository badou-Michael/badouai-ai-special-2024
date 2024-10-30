import numpy as np
from numpy import random


def pca(X, k):
    # 1. 中心化数据
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # 2. 计算协方差矩阵
    cov_matrix = np.cov(X_centered.T)

    # 3. 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # 4. 选择前k个最大特征值对应的特征向量
    top_k_indices = np.argsort(eigenvalues)[-k:]
    top_k_eigenvectors = eigenvectors[:, top_k_indices]

    # 5. 将原始数据投影到新的低维空间
    X_reduced = np.dot(X_centered, top_k_eigenvectors)

    return X_reduced

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
                  [9,  48, 12, 5],
                  [11, 21, 23, 14],
                  [8,  5, 45, 15],
                  [11, 12, 1, 21],
                  [21, 20, 12, 25]])
    # 调用pca函数，降维到2维
    X_reduced = pca(X, 3)
    print("降维后的矩阵：")
    print(X_reduced)
