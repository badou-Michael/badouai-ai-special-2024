from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt


X = [[2, 9], [8, 5], [7, 4], [2, 2], [4, 1]]
Z_single = linkage(X, 'single')
f_single = fcluster(Z_single, 4, criterion='distance')
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z_single)
plt.title("Single Linkage")
plt.show()
print("Single Linkage Result:", f_single)
