from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
from matplotlib import pyplot as plt

X=[[1,2],[3,4],[4,4],[1,2],[1,3]]

Z=linkage(X,'ward')
f=plt.figure(figsize=(5,3))
dn=dendrogram(Z)
print(Z)
plt.show()
