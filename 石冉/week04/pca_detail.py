import numpy as np

#定义pca类
class PCA(object):
    '''PCA方法'''
    def __init__(self,X,K): #传入样本矩阵X，及需降维至几阶的阶数k
        self.X=X
        self.K=K
        self.centrX=[] #将中心化矩阵赋值为空
        self.C=[] #将协方差矩阵赋值为空
        self.U=[] #将X的转换矩阵赋值为空
        self.R=[] #将X的降维矩阵赋值为空

        self.centrX=self.centralized() #调用centralized()函数，将X中心化
        self.C=self.cov() #调用cov()函数，求协方差矩阵
        self.U=self.trans() #调用trans()方法，求X的转换矩阵
        self.R=self.result() #调用result()方法，求X的降维矩阵R=XU


    def centralized(self):
        '''中心化处理'''
        mean=np.array([np.mean(i) for i in self.X.T]) #求X的每一列（遍历每个特征中所有值）的均值，并以数组形式返回
        centrX=[]
        centrX=self.X-mean #X减去均值，得到均值为0
        return centrX

    def cov(self):
        '''求协方差矩阵'''
        m=np.shape(self.centrX)[0] #获取centrX的行数
        C=np.dot(self.X.T,self.X)/(m-1) #计算协方差矩阵，除以m-1是无偏估计
        return C

    def trans(self):
        '''求转换矩阵'''
        a,b=np.linalg.eig(self.C)#求特征值和特征向量，特征值赋给a，特征向量赋给b
        ind=np.argsort(-1*a)#将特征值按从大到小的顺序排列
        UT=[b[:,ind[i]] for i in range(self.K)]#将ind作为索引，取前k个重要特征向量，构建转换矩阵
        U=np.transpose(UT)
        return U

    def result(self):
        '''计算降维矩阵'''
        R=np.dot(self.X,self.U)
        print(R)
        return R



# 生成10个样本，每个样本有3个特征，数据范围从0到100
X = np.random.randint(0, 101, size=(10, 3))
print(X)
K=np.shape(X)[1]-1
pca=PCA(X,K)

