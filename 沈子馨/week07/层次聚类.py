from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

X = [[1,2],[3,2],[4,4],[1,2],[1,3]]
Z = linkage(X, 'ward')               #linkage(y, method=’single’, metric=’euclidean’) 
f = fcluster(Z,4,'distance')         #fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None)
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()
