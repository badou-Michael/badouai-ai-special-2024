import numpy as np

class CPCA(object):
    def __init__(self, X, K):
        # 等号左边的都叫类属性，类内所有函数都可以调用赋值
        self.X = X
        self.K = K
        # self.centralX = []
        # self.C = []
        # self.U = []
        # self.Z = []
        self.centralX = self._centralized()     # 调用_centralized  并将返回值赋给等号左侧
        self.C = self._Cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):     # 方法名或属性名前加单下划线_用于表示该属性或方法是“受保护的”或“内部使用的”
        # 求特征均值
        mean = np.array([np.mean(i) for i in self.X.T])       # 列表推导式
        print('特征均值为：\n', mean)
        # 对样本矩阵做中心化处理
        centralX = self.X - mean        # self.X——调用类属性；centralX是函数内的变量对象，只能在函数内部使用（函数运行完就没了）
        print('中心化矩阵为：\n', centralX)
        return centralX

    def _Cov(self):
        # 对中心化矩阵求协方差矩阵→求中心化矩阵的行数
        n = np.shape(self.centralX)[0]
        C = np.dot(self.centralX.T, self.centralX) / (n-1)
        print('协方差矩阵为：\n', C)
        return C

    def _U(self):
        # 求降维转换矩阵→求特征值和特征向量
        w, v = np.linalg.eig(self.C)
        print('特征值为：\n', w)
        print('特征向量为：\n', v)
        # 从大到小对特征值进行排序
        ind = np.argsort(-1*w)
        # 按降维数量K取出前K个特征值所对应的特征向量构成降维转换矩阵
        UT = [v[:, ind[i]] for i in range(self.K)]      # 列表推导式
        print('-----:', UT)     # 输出的UT为数组array形式
        U = np.transpose(UT)       # 返回数组的转置
        # self.U = np.array(UT).T   # UT是一个列表对象，需要先转换成数组对象再进行转置；直接UT.T会产生语法报错
        print('降维转换矩阵为：', U)
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('降维矩阵为：', Z)
        return Z

if __name__ == "__main__":
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1]-1
    pca = CPCA(X, K)
