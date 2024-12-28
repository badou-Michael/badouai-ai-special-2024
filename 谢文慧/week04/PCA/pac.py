# PCA函数详细
# 使用PCA求样本矩阵X的K阶降维矩阵Z
import numpy as np
class PCA(object):
    # 用PCA求样本矩阵X的K阶降维矩阵Z
    # 输入的样本矩阵X  shape=(m,n),m行样例，n个特征
    def __init__(self,X,K):
        self.X=X  #样本矩阵X
        self.K=K  #K阶降维矩阵的K值
        self.centX = []  #矩阵X的中心化
        self.C=[]  # 样本集的协方差矩阵C
        self.U=[]  # 样本矩阵X的降维转换矩阵
        self.Z=[]  # 样本矩阵X的降维矩阵Z

        self.centX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z() 
    def _centralized(self):
        # 矩阵X的中心化
        print('样本矩阵X：\n',self.X)
        centX = []
        # 样本集的特征均值
        mean = np.array([
            np.mean(attr) for attr in self.X.T
        ])
        print('样本集的特征均值：\n',mean)
        centX = self.X-mean #样本集的中心化
        print('样本矩阵X的中心化centX:\n',centX)
        return centX
        
    def _cov(self):
        # 求样本矩阵X的协方差矩阵C
        # 样本集的样例总数
        num = np.shape(self.centX)[0]
        # 样本矩阵的协方差矩阵C
        C = np.dot(self.centX.T,self.centX)/(num-1)
        print('样本矩阵X的协方差矩阵C：\n',C)
        return C
        
    def _U(self):
#       求X的降维转换矩阵U，shape=(n,k),n是X的特征维度总数，k是降维矩阵的特征维度
    # 先求X的协方差矩阵C的特征值和特征向量
        a,b = np.linalg.eig(self.C)
        print('协方差矩阵C的特征值：\n',a)
        print('协方差矩阵C的特征向量：\n',b)
        # 给出特征值降序的topK的索引序列
        ind = np.argsort(-1*a)
        # 构建K阶降维的降维转换矩阵U
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U：\n'%self.K,U)
        return U
        
    def _Z(self):
        # 按照Z=XU求降维矩阵Z，shape=(m,k),n是样本总数，k是降维矩阵中特征维度总数
        Z = np.dot(self.X,self.U)
        print('X shape',np.shape(self.X))
        print('U shape',np.shape(self.U))
        print('Z shape',np.shape(self.Z))
        print('样本矩阵X的降维矩阵Z：\n',Z)
        return Z

if __name__=='__main__':
    X = np.array(
        [
            [12,34,34],
            [15,32,44],
            [6,25,56],
            [62,24,74],
            [7,18,38],
            [42,23,32],
            [52,77,31],
            [22,55,21],
            [61,44,3],
            [35,65,5]
        ]
    )
    K = np.shape(X)[1]-1
    print('样本集（10行3列，10个样例，每个样例3个特征）：\n',X)
    pca = PCA(X,K)
    
