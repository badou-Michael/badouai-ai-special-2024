import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from tensorflow import float32, float64


def hierarchy_clustering(x: np.array):
    # 样本数，特征数
    n_samples, n_features = x.shape
    x = [[point] for point in x]
    # 标签
    labels = np.arange(n_samples)
    idx = n_samples
    res = []
    while len(np.unique(labels)) > 1:
        # 距离矩阵
        distances = np.zeros((len(labels), len(labels)))
        for i in range(len(labels)):
            distances[i, i] = np.inf
            for j in range(i + 1, len(labels)):
                # 质心的距离
                distances[i][j] = np.sqrt(sum((np.mean(x[labels[i]], axis=0) - np.mean(x[labels[j]], axis=0))**2))
                distances[j, i] = distances[i, j]

        # 找到最小值的索引（扁平化后）
        min_index_flat = np.argmin(distances)
        rows = distances.shape[0]
        min_row, min_col = min_index_flat // rows, min_index_flat % rows
        res.append([labels[min_row], labels[min_col], distances[min_row, min_col], len(x[labels[min_row]] + x[labels[min_col]])])
        x.append(x[labels[min_row]] + x[labels[min_col]])
        print(labels)
        labels = labels[(labels != labels[min_row]) == (labels != labels[min_col])]
        labels = np.append(labels, idx)
        idx += 1
        # print(distances)
        print(labels)
        print(min_row, min_col)
    return res


# 假设X是数据点，labels是聚类结果
X = np.array([[1, 2], [1, 4], [1, 9],
              [4, 2], [4, 9], [4, 0]])

# 计算距离矩阵并进行层次聚类
Z = hierarchy_clustering(X)
# Z = linkage(X, 'ward') # 使用Ward方法
print(Z)
J = [[0, 3, 0, 2], [4, 5, 1.15, 3], [1, 2, 2.23, 2], [6, 7, 4.00, 5]]
# 画出树状图
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()
