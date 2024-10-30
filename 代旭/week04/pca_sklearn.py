import numpy as np
from sklearn.decomposition import PCA

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
pca = PCA(n_components=2)
pca.fit(X)
newX = pca.fit_transform(X)
print(newX)
