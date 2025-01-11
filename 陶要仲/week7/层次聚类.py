###cluster.py
#导入相应的包
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt


X = [[1, 2], [3, 2], [1, 4], [2, 3], [4, 5]]
Z = linkage(X, 'ward')
f = fcluster(Z, 3, 'distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
print(dn)
print(f)
plt.show()

