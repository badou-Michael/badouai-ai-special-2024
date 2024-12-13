from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
import numpy as np
from matplotlib import pyplot as plt

x = np.random.rand(10)
x = x.reshape((5,2))
print(x)
z = linkage(x, 'ward')
dn = dendrogram(z)
clusters = fcluster(z, 0.5, 'distance')
print(clusters)
plt.scatter(x[:,0],x[:,1],c = clusters,cmap='rainbow')
plt.show()