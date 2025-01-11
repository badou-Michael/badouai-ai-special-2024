import numpy as np
from sklearn.decomposition import PCA

class PCA(object):
    def __init__(self,X,K):
        self.X=X
        self.K=K


        self.centX=self._cent()
        self.C=self._cov()
        self.W=self._eigenvector()
        self.X_new=self._X_new()

    def _cent(self):#mean for each dimension(每个特征的均值）
        x_bar=np.array([np.mean(i) for i in self.X.T])
        centX=self.X-x_bar
        return centX

    def _cov(self):
        C=np.dot(self.centX.T,self.centX)/np.shape((self.X)[0]-1)
        return C

    def _eigenvector(self):
        evalue,evector=np.linalg.eig(self.C)
        index=np.argsort(-evalue)
        #print(self.K)
        W=evector[:,:self.K]
        #print(W)
        return W


    def _X_new(self):
        X_new=np.dot(self.centX,self.W)
        print("降为矩阵：\n",X_new,"\n降维值K为",self.K)
        return X_new


if __name__=="__main__":
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
    K=np.shape(X)[1]-1
    PCA(X,K)
            


'''
接口调用
pca = PCA(n_components=2)
pca.fit(X)
newX=pca.fit_transform(X)
print(newX)
'''
