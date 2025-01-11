import numpy as np
from sklearn.decomposition import PCA

samples = np.random.randint(0,100,size=(10,3))
print(samples)
pca=PCA(n_components=2)
#pca.fit(samples) #仅拟合
new_samples=pca.fit_transform(samples) #拟合加转换
print(new_samples)
