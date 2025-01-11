import numpy as np

class PCA(object):
    def __init__(self, Z, K):
        """
        :param Z: 样本矩阵
        :param K: 降成的维度
        """
        self.Z = Z
        self.K = K
        self.CentraZ = self._Centralization()
        self.CovZ = self._Cov()
        self.valuesZ, self.vectorsZ = self._ED()
        self.W = self._PC()#主成分矩阵
        self.U = self._PJ()#降维以后的矩阵

    def _Centralization(self):
        """
        :return:标准化以后的数据
        """
        #计算每一个特征的均值和标准差
        mean = np.mean(self.Z, axis=0)#每一列的均值
        std = np.std(self.Z, axis=0)#每一列的标准差

        #标准化数据矩阵
        Z_std = (self.Z - mean)/std

        return Z_std

    def _Cov(self):
        """
        :return: 协方差矩阵
        """

        #样本的数量
        m = np.shape(self.CentraZ)[0]

        #计算协方差
        Cov = np.dot(self.CentraZ.T,self.CentraZ)/m

        return Cov

    def _ED(self):
        """
        :return:协方差矩阵的特征值和特征向量
        """
        values, vectors = np.linalg.eig(self._Cov())

        return values, vectors

    def _PC(self):
        """
        :return: 构成主成分矩阵
        """
        indices = np.argsort(self.valuesZ)[::-1][:self.K]

        W = self.vectorsZ[:, indices]

        return W

    def _PJ(self):
        """
        :return:原始数据投影到主成分空间
        """
        #X = ZW

        X = np.dot(self.Z, self.W)

        print("样本的降维矩阵为:\n", X)

        return X


if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    Z = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(Z)[1]-1
    pca = PCA(Z,K)
