import numpy as np
from sklearn.decomposition import PCA
X = np.array([[12, 18, 27],
              [14, 42, 17],
              [22, 19, 33],
              [13, 10, 38],
              [40, 50, 15],
              [7,  49, 8],
              [10, 23, 16],
              [6,  7,  12],
              [10, 14, 20],
              [24, 22, 28]])
pca = PCA(n_components=2)   #降到2维
pca.fit(X)                  #执行
newX=pca.fit_transform(X)   #降维后的数据
print(newX)                  #输出降维后的数据
