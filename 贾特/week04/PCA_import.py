import  numpy as np
from sklearn.decomposition import PCA

X = np.array([[20,2,33],
                 [12,23,45],
                 [22,34,66],
                  [32,66,45],
                  [23,65,43],
                  [2,2,5],
                  [33,24,67],
                  [21,64,23],
                  [33,55,87],
                  [90,98,36]])
pca = PCA(n_components=2)   #声明
pca.fit(X)                  #执行，协方差矩阵求解
newX = pca.fit_transform(X)  #降维求解后的数据
print(newX)
