import numpy as np

# ndarray shape: m,n  m个sample， n个label
# k：目标维度
def pca(ndarray, k):
    ndarray = ndarray - ndarray.mean(axis=0)
    covariance = np.dot(ndarray.T, ndarray)/ndarray.shape[0]
    vals, vecs = np.linalg.eig(covariance)
    target_val = np.argsort(-1*vals)
    target_vec = vecs[:, target_val[:k]]    # 转换矩阵
    return np.dot(ndarray, target_vec)

ndarray = np.random.randint(0,100,size=(8,3))

print(pca(ndarray, 2))