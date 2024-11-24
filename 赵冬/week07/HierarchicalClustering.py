import math

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt

# c1,c2是序号集合
def distance(X, c1, c2, method):
    if method == 'single':
        dists = np.array([[np.linalg.norm(X[c1[i]] - X[c2[j]]) for i in range(len(c1))] for j in range(len(c2))])
        return np.min(dists)

X = np.array([[1,2],[3,2],[4,4],[1,2],[1,3]])
Z = []
# 保存每个合并序号包含的子集
merge_index = [[i] for i in range(len(X))]
# 每步合并剩余未合并的序号
un_merge = np.array([i for i in range(len(X))])
while len(un_merge) > 1:
    distances = np.array([[distance(X, merge_index[un_merge[i]], merge_index[un_merge[j]], 'single') for i in range(0, len(un_merge))] for j in range(0, len(un_merge))])
    np.fill_diagonal(distances, np.inf)
    flat_index = np.argmin(distances)
    coords = np.unravel_index(flat_index, distances.shape)
    merge_index.append([])
    Z = Z +[[un_merge[coords[0]],un_merge[coords[1]], np.min(distances), len(merge_index[un_merge[coords[0]]])+len(merge_index[un_merge[coords[1]]])]]
    for i in range(len(coords)):
        merge_index[-1] = merge_index[-1] + merge_index[un_merge[coords[i]]]
    un_merge = np.delete(un_merge, coords, axis=0)
    un_merge = np.append(un_merge, len(merge_index) - 1)
fig = plt.figure(figsize=(5, 3))
dn = dendrogram(Z)
print(Z)
plt.show()


