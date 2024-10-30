import numpy as np
from sklearn.decomposition import PCA

X = np.array(
    [[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2], [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
pca = PCA(n_components=2)  # 降到2维
pca.fit(X)  # 执行,计算数据的协方差矩阵并获取特征值和特征向量。
newX = pca.fit_transform(X)  # 降维后的数据，fit和transform的结合，输出降维后的数据
print(newX)  # 输出降维后的数据
