from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
import numpy as np

X=np.zeros((5,2))

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        n=10*np.random.random()
        X[i,j]=n

Z = linkage(X, 'ward')    
f = fcluster(Z,4,'distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
print(X.shape)
plt.show()
