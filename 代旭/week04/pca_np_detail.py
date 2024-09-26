import numpy as np
class PCA():
    def __init__(self,n_components):
        self.n_components = n_components

    def fit_transform(self,X):
        self.n_features_ = X.shape[1]
        X=X-X.mean(axis=0)
        self.covariance = np.dot(X.T,X)/X.shape[0]
        eig_vals,eig_vectors = np.linalg.eig(self.covariance)
        idx = np.argsort(-eig_vals)
        self.n_components_=eig_vectors[:,idx[:self.n_components]]
        return np.dot(X,self.n_components_)

pca = PCA(n_components=2)
X=np.array([[1, 2, 3],
                  [2, 3, 4],
                  [3, 4, 5],
                  [4, 5, 6],
                  [5, 6, 7],
                  [6, 7, 8],
                  [7, 8, 9],
                  [8, 9, 10],
                  [9, 10, 11],
                  [10, 11, 12]])
newX = pca.fit_transform(X)
print(newX)
