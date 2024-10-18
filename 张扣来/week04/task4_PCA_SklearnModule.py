import numpy as np
from sklearn.decomposition import PCA
X = np.array([
            [12,16,25],
            [11,29,78],
            [28,18,38],
            [25,29,37],
            [39,49,56],
            [29,28,29],
            [18,47,65],
            [49,25,37]
            ])
pca = PCA(n_components=2)
pca.fit(X)
# transform转换
Y = pca.transform(X)
print(Y)