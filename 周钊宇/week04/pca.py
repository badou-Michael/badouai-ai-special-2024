import numpy as np

class PCA(object):
    
    def __init__(self, X, K):
        self.X = X #输入的样本矩阵
        self.K = K #需要的降维矩阵维度
        self.centralX = self.__centralized() #中心化矩阵
        self.D = self.__conv() #协方差矩阵
        self.U = self.__U() #降维转换矩阵   
        self.Z = self.__Z() #降维特征矩阵    
         
    def __centralized(self):
        
        """中心化"""
        
        print("样本矩阵：\n", self.X)
        mean = np.array([np.mean(line) for line in self.X.T])
        print("样本均值：\n",mean)
        centralX = self.X-mean
        print("中心化矩阵：\n",centralX)
        return centralX
    
    def __conv(self):
        """计算中心化之后矩阵的协方差矩阵"""
        print("中心化样本矩阵的维度:\n",self.X.shape[0])
        D = np.dot(self.centralX.T, self.centralX) / (self.X.shape[0]-1)
        print("协方差矩阵:\n",D)         
        return D
    
    def __U(self):
        """计算协方差矩阵的特征值和特征向量，求降维转换矩阵U"""
        a,b = np.linalg.eig(self.D) #a是特征值，b是特征向量
        print("协方差矩阵的特征值是:\n",a)
        print("协方差矩阵的特征向量是:\n",b)
        suoyin = np.argsort(-1*a)
        UT  =  np.array([b[:,suoyin[i]] for i in range(self.K)])
        print("降维转换矩阵U的逆的维度:\n",UT.shape)
        U = UT.T
        print("降维转换矩阵U:\n",U)
        print("降维转换矩阵U的维度:\n",U.shape)
        return U
    
    def __Z(self):
        """Z = XU 求降维之后的特征矩阵"""
        Z = np.dot(self.X, self.U)
        print("降维之后的样本矩阵:\n", Z)
        return Z
        
if __name__=='__main__':
    
    X = np.random.randint(0,10,(10,3))
    # X = np.array([[10, 15, 29],
    #               [15, 46, 13],
    #               [23, 21, 30],
    #               [11, 9,  35],
    #               [42, 45, 11],
    #               [9,  48, 5],
    #               [11, 21, 14],
    #               [8,  5,  15],
    #               [11, 12, 21],
    #               [21, 20, 25]])
    res = PCA(X,2)
    
        