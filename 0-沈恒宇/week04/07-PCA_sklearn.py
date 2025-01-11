'''
使用sklearn中的pca接口
1.导入模块
2.创建对象，设置参数
4.得到降维后的数据
5.打印
'''
import numpy as np
from sklearn.decomposition import PCA


X = np.array([[-1,2,66,-1], [-2,6,58,-1], [-3,8,45,-2], [1,9,36,1], [2,10,62,1], [3,5,83,2]])  #导入数据，维度为4
pca = PCA(n_components=2)
new_X = pca.fit_transform(X)
print(new_X)