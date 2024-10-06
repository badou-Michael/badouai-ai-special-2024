import numpy
import random

class My_PCA(object):
    def __init__(self,matrix,k):
        self.matrix = matrix       # 源矩阵
        self.k = k                 # 降维维度
        self.matrix_center = []    # 中心化矩阵
        self.matrix_C = []         # 协方差矩阵
        self.matrix_U = []         # 协方差降维矩阵
        self.Z = []                # 降维矩阵

        self.matrix_center = self.center()
        self.matrix_C = self.cov()
        self.matrix_U = self.val()
        self.Z = self.result()

    def center(self):              # 中心化矩阵
        print('源矩阵：\n',self.matrix)
        mean = self.matrix.mean(axis=0)
        print('特征均值：\n',mean)
        matrix_center = self.matrix - mean
        print('中心化矩阵：\n',matrix_center)
        return matrix_center

    def cov(self):                 # 协方差矩阵
        n = self.matrix_center.shape[0]
        matrix_C = numpy.dot(self.matrix_center.T,self.matrix_center) / ( n - 1 )
        print('协方差矩阵：\n',matrix_C)
        return matrix_C

    def val(self):                 # 协方差降维矩阵
        vals, vecs = numpy.linalg.eig(self.matrix_C)
        print('协方差矩阵C的特征值:\n', vals)
        print('协方差矩阵C的特征向量:\n', vecs)
        ind = numpy.argsort(-1*vals)
        v = [vecs[:,ind[i]] for i in range(self.k)]
        U = numpy.transpose(v)
        print('协方差矩阵C的降维矩阵:\n', U)
        return U

    def result(self):
        Z = numpy.dot(self.matrix,self.matrix_U)
        print('降维矩阵:\n', Z)
        return Z

if __name__ == '__main__':
    # matrix = numpy.array( [[random.randrange(0,255) for _ in range(5)] for _ in range(10)] )
    X = numpy.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    test = My_PCA(X,2)
