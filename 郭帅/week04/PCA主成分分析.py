#pca主成分分析

import numpy as np

def pca(X, n_components):
    # 1. 数据中心化：减去每个特征的均值
    mean = np.mean(X, axis=0)  # 计算每个特征的均值
    X_centered = X - mean  # 从每个特征减去其均值，得到中心化后的数据

    print("中心化后的数据：\n", X_centered)

    # 2. 计算协方差矩阵
    cov_matrix = np.cov(X_centered, rowvar=False)  # 协方差矩阵
    print("协方差矩阵：\n", cov_matrix)

    # 3. 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # 使用eigh更适合对称矩阵
    print("特征值：\n", eigenvalues)
    print("特征向量：\n", eigenvectors)

    # 4. 排序特征值，并选择前n_components个特征向量
    sorted_indices = np.argsort(eigenvalues)[::-1]  # 特征值从大到小排序
    eigenvectors_sorted = eigenvectors[:, sorted_indices[:n_components]]  # 选择前n个特征向量
    print(f"前{n_components}个特征向量：\n", eigenvectors_sorted)

    # 5. 将数据投影到选定的特征向量上，得到降维后的数据
    X_reduced = np.dot(X_centered, eigenvectors_sorted)  # 将数据投影到主成分上
    print("降维后的数据：\n", X_reduced)

    return X_reduced


# 随机生成 10 个样本，每个样本有 5 个特征
X = np.random.rand(10, 5)

# 降维到 2 个主成分
X_reduced = pca(X, 2)
