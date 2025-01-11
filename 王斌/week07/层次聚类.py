from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
from sklearn import datasets


iris = datasets.load_iris()
X = iris.data[:, :4]
Z = linkage(X, 'ward')
f = fcluster(Z,4,'distance')
fig = plt.figure(figsize=(8, 8))
dn = dendrogram(Z)
print(Z)
plt.show()
