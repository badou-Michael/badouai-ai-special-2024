
from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
import matplotlib.pyplot as plt

X = [[1,0],[4,8],[4,2],[1,2],[1,3],[2,3],[3,4]]
Z = linkage(X, 'ward')
f = fcluster(Z,4,'distance')
fig = plt.figure(figsize=(10,6))
dn = dendrogram(Z)
print(Z)
plt.show()
