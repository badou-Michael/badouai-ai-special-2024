import numpy as np

# 定义PCA函数
def pca(ndarray, k):
    # 中心化
    ndarray = ndarray - ndarray.mean(axis=0)
    # 计算协方差矩阵
    covariance = np.dot(ndarray.T, ndarray) / ndarray.shape[0]
    # 计算特征值和特征向量
    vals, vecs = np.linalg.eig(covariance)
    # 按照特征值降序排列，取得对应的特征向量
    target_val = np.argsort(-1 * vals)
    target_vec = vecs[:, target_val[:k]]
    # 将原始数据映射到新的子空间
    return np.dot(ndarray, target_vec)

# 生成一个8x3的随机整数数组
ndarray = np.random.randint(0, 50, size=(5, 5))
# 打印原矩阵结果
print(ndarray)

# 执行PCA，目标维度为2
result = pca(ndarray, 2)

# 打印PCA结果
print(result)