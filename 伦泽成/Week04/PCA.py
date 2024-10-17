#PCA numpy
import numpy as np
class PCA():
    def __init__(self,n_components):
        self.n_components = n_components
    
    def fit_transform(self,X):
        self.n_features_ = X.shape[1]
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T,X)/X.shape[0]
        eig_vals,eig_vectors = np.linalg.eig(self.covariance)
        idx = np.argsort(-eig_vals)
        self.components_ = eig_vectors[:,idx[:self.n_components]]
        return np.dot(X,self.components_)
 
pca = PCA(n_components=2)
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]]) 
newX=pca.fit_transform(X)
print(newX)                 



#PCA numpy details
使用PCA求样本矩阵X的K阶降维矩阵Z
import numpy as np
 
class CPCA(object):
    def __init__(self, X, K):
        self.X = X       #样本矩阵X
        self.K = K       #K阶降维矩阵的K值
        self.centrX = [] #矩阵X的中心化
        self.C = []      #样本集的协方差矩阵C
        self.U = []      #样本矩阵X的降维转换矩阵
        self.Z = []      #样本矩阵X的降维矩阵Z
        
        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z() #Z=XU求得
        
    def _centralized(self):
        print('样本矩阵X:\n', self.X)
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T]) #样本集的特征均值
        print('样本集的特征均值:\n',mean)
        centrX = self.X - mean ##样本集的中心化
        print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX
        
    def _cov(self):
        ns = np.shape(self.centrX)[0]
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C
        
    def _U(self):
        a,b = np.linalg.eig(self.C) #特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html 
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        ind = np.argsort(-1*a)
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n'%self.K, U)
        return U
        
    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z
        
if __name__=='__main__':
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
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = CPCA(X,K)


#PCA sklearn
import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]]) 
pca = PCA(n_components=2)   #降到2维
pca.fit(X)                  #执行
newX=pca.fit_transform(X)   #降维后的数据
print(newX)                  #输出降维后的数据



#PCA iris
import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets.base import load_iris

x,y=load_iris(return_X_y=True) #加载数据，x表示数据集中的属性数据，y表示数据标签
pca=dp.PCA(n_components=2) #加载pca算法，设置降维后主成分数目为2
reduced_x=pca.fit_transform(x) #对原始数据进行降维，保存在reduced_x中

red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]
for i in range(len(reduced_x)): 
    if y[i]==0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i]==1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
plt.scatter(red_x,red_y,c='r',marker='x')
plt.scatter(blue_x,blue_y,c='b',marker='D')
plt.scatter(green_x,green_y,c='g',marker='.')
plt.show()
