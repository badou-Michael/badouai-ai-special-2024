import numpy as np
from sklearn.decomposition import PCA
samples = np.random.randint(0,100,size=(10,3))
pca = PCA(n_components=2)   #降到2维
pca.fit(samples)                  #执行
new_samples=pca.fit_transform(samples)   #降维后的数据
print(new_samples)
