import cv2
import numpy as np


class CPCA:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.center_mean = []
        self.c = []  # 样本协方差矩阵
        self.u = []  # 样本降维转换矩阵
        self.z = []  # 样本降维矩阵

        self.center_mean = self.compute_center_mean()
        self.c = self.compute_c()
        self.u = self.compute_u()
        self.z = self.compute_z()

    def compute_center_mean(self):
        # 中心化
        # mean = []
        # for i in self.data.T:
        #     mean.append(np.mean(i))
        centerX = []
        mean = np.array([np.mean(i) for i in self.data.T])
        centerX = self.data - mean
        print(mean, centerX, "centerX")
        return centerX

    def compute_c(self):
        # 协方差
        # print(self.center_mean)
        num = np.shape(self.center_mean)[0]
        print(num,"样本数")
        c_c = np.dot(self.center_mean.T, self.center_mean) / num
        print(num, c_c, "c_c")
        return c_c

    def compute_u(self):
        a, b = np.linalg.eig(self.c)  # a特征值，b特征向量
        print(a, b, "特征值/特征向量")
        # 特征值倒序
        sort = np.argsort(-1 * a)
        ut = [b[:, sort[i]] for i in range(self.k)]
        print(ut,sort, "ut")
        u = np.transpose(ut)
        print(u)
        return u

    def compute_z(self):
        c_z = np.dot(self.data, self.u)
        print(c_z, 'c_z')
        return c_z


if __name__ == '__main__':
    x = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
    k = x.shape[1] - 1
    pca = CPCA(x, k)
    # pca = pca_fun(x, k)
