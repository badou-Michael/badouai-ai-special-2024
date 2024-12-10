#利用sklearn库来快速实现PCA

import numpy as np
from sklearn.decomposition import PCA

#导入一个四维数据
X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])
#降到二维
pca = PCA(n_components=2)
pca.fit(X)  #执行
newX = pca.fit_transform(X) #降维后的数据
print(newX)
