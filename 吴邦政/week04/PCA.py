import numpy as np

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        self.n_features_ = X.shape[1]
        X = X - X.mean(axis=0)
        self.covariance = np.dot(X.T, X) / X.shape[0]
        eig_vals, eig_vectors = np.linalg.eig(self.covariance)
        idx = np.argsort(-eig_vals)
        self.components_ = eig_vectors[:, idx[:self.n_components]]
        return np.dot(X, self.components_)


pca = PCA(n_components=2)
X = np.array(
    [[-8,79,6,44], [65,87,-9,-3], [54,-5,-4,-66], [22,54,-54,-3], [41,54,23,-21], [666,888,777,333]])
newX = pca.fit_transform(X)
print(newX)
