import numpy as np
from tensorflow.compiler.tf2xla.python.xla import self_adjoint_eig


class CPCA(object):
    # 特殊构造函数，指代实例本身__int__(self)
    def __init__(self, X, K):
        self.X = X #样本矩阵数据
        self.K = K #X的降维矩阵的阶数，即X要特征降维成k阶
        self.centerX = [] #矩阵X去中心化
        self.C = []  #样本集的协方差矩阵
        self.U = []  #矩阵X降维转换矩阵
        self.Z = []  #矩阵X的降维举证Z
        self.centerX = self._centralized()#求centerX：中心化过的数组
        self.C = self._cov() #C:协方差矩阵
        self.U = self._U()#U：降维转换矩阵
        self.Z = self._Z() #Z:将X进行PPCA降维之后的矩阵

    def _centralized(self):
        print("样本矩阵\n",self.X)
        # centerX = []
        # 样本集均值,np.arry()创建一个数组，np.mean创建均值，对X的每一列数据求均值
        mean = np.array([np.mean(attr) for attr in self.X.T])
        centerX = self.X-mean
        print("样本集减去均值后，去中心化数据：\n",centerX)
        return centerX
    def _cov(self):
        # 样本集样本实例总数
        ns = np.shape(self.centerX)[0]
        C = np.dot(self.X.T,self.centerX)/(ns-1)# 求样本的协方差矩阵C：这里套用做完中心化之后的公式
        print("样本矩阵的协方差矩阵：\n",C)
        return C
    def _U(self):
        # 求C的特征值和特征向量
        # 特征值赋值给a，对应特征向量赋值给b
        a, b = np.linalg.eig(self.C)
        print("样本矩阵的协方差C的特征值：\n", a)
        print("样本矩阵的协方差C的特征向量值：\n", b)
        ind = np.argsort(-1 * a)#给出特征值的降序的topk索引序列，这个很重要
        #构建K阶降维的降维转换矩阵U，协方差矩阵求特征值b,Ax = λx，x作为λ特征向量对应b，λ作为特征值对应a
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print("%d阶降维转换矩阵为：\n" %self.K,U)
        return U

    def _Z(self):
        Z = np.dot(self.X,self.U)
        print("X shape:",np.shape(self.X))
        print("U shape:",np.shape(self.U))
        print("Z shape:",np.shape(Z))
        print("样本X的降维矩阵Z:\n",Z)
        return Z
if __name__ == "__main__":
    X = np.array([[12,18,25],[23,24,28],[13,25,35],[10,20,42],[76,28,50],
                 [19,27,68],[49,16,18],[28,27,65],[10,37,12],[16,23,56]])
    K = np.shape(X)[1]-1
    print ("先定义样本集，10个样本，10行3列，3个特征\n",X)
    pca = CPCA(X, K)