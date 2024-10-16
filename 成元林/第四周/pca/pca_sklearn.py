import numpy as np
from sklearn.decomposition import PCA

X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])
# 降到2维设定
pca = PCA(n_components=2)
# 执行降维
fit = pca.fit(X)
newX = pca.fit_transform(X)
print(newX)