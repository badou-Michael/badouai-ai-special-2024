'''
用PCA算法求样本数据矩阵X的K阶降维矩阵Z
'''

import numpy as np



class PCA(object):

    #初始化
    def __init__(self, X,k):
        self.X = X
        self.k = k

        #样本矩阵的中心化矩阵A
        self.A = self.center_matrix()
        #样本矩阵的协方差矩阵B
        self.B = self.x_matrix()
        #k阶转换矩阵C
        self.C = self.x1_x2()
        #样本集的k阶降维矩阵
        self._Z = self.end_Z()
    '''
    零均值化矩阵A
    样本数据矩阵的每一行是一个数据，每一列是一个维度，计算均值按维度计算。
    此处需要矩阵转置方便计算 
    
    此处的一个问题：self.X - mean，两个矩阵应该（行和列）相等才能做加减操作
    mean是一个一维数组，mean应该自动扩容与self.X相同，
    '''
    def center_matrix(self):
        centerX = []
        mean = np.array([np.mean(temp) for temp in self.X.T])   #列表推导式
        print('样本矩阵X的特征均值\n',mean)
        centerX = self.X - mean
        print('样本矩阵X的零均值化（中心化）矩阵\n',centerX)
        return centerX


    '''
    求协方差矩阵B
    根据中心化的协方差矩阵公式 D=z(T)z/m
    np.dot()库函数是计算连个数组的点积（对应元素相乘再求和）
    '''
    def x_matrix(self):
        #样本总数
        m = self.A.shape[0]
        B = np.dot(self.A.T,self.A )/m
        print('样本矩阵的协方差矩阵\n',B)
        return B

    '''
    求协方差矩阵的特征值和特征向量 ，由前两个值构建一个k阶降维矩阵
    np.linalg.eig()
    '''
    def x1_x2(self):
        #求x1,x2
        x1,x2 = np.linalg.eig(self.B)
        print('协方差矩阵的特征值\n',x1)
        print('协方差矩阵的特征向量\n',x2)
        #对特征值取降序索引
        index = np.argsort(x1)
        #特征值对应的列向量组合矩阵
        C_T = [x2[:,index[i]] for i in range(self.k)]
        #转置
        C = np.transpose(C_T)
        print('%d阶降维转换矩阵C\n'%self.k,C)
        return C



    '''
    求降维矩阵  = 样本矩阵x k阶降维矩阵   
    '''
    def end_Z(self):
        Z = np.dot(self.X,self.C)
        print('X space:',np.shape(self.X))
        print('C space:',np.shape(self.C))
        print('Z space:',np.shape(Z))
        print('样本集的k阶降维矩阵Z\n',Z)


if __name__ == '__main__':
    #样本数据集 ，行为样本，列为维度
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])

    #降维维度K(3-1)
    K = X.shape[1]-1
    print('样本数据集，10x3,\n',X)
    PCA(X,K)
