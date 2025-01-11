from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
from matplotlib import pyplot as plt

X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
Y = linkage(X,'average')
f = fcluster(Y, 4, "distance")
fig = plt.figure(figsize=(5,3))
dn = dendrogram(Y)
print(Y)
plt.show()