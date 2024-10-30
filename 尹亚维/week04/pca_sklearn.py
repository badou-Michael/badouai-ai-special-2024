import numpy as np
from sklearn.decomposition import PCA
# 导入数据，维度为4
X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])
# 实例化PCA,并降维到2维
pca = PCA(n_components=2)
# 训练
pca.fit(X)
# 转换
newX = pca.transform(X)
print(newX)
