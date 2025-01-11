import cv2
import numpy as np


class PCA:
    def __init__(self, k):
        self.k = k

    def fit_transform(self, data):
        data = data - data.mean(axis=0)
        print(data, "中心化")
        x = np.dot(data.T, data) / (data.shape[0])
        print(x, "协方差")
        a, b = np.linalg.eig(x)
        print(a, b, "特征值。特征向量")
        sort = np.argsort(-a)
        c = [b[:, sort[i]] for i in range(self.k)]
        z = np.transpose(c)
        print(z,data,"降维矩阵")
        return np.dot(data, z)


pca = PCA(k=2)
x = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]]);
newx = pca.fit_transform(x)
print(newx)
