# 1. 从头实现K-Means算法
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(42)
data = np.random.rand(100, 2)

def initialize_centroids(data, k):
    # 随机初始化k个中心点
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    # 根据欧氏距离将每个点分配到最近的中心点
    distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(data, labels, k):
    # 更新每个簇的中心点为簇中所有点的平均值
    new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans(data, k, max_iters=100):
    # 初始化中心点
    centroids = initialize_centroids(data, k)
    for i in range(max_iters):
        # 分配每个点到最近的中心点
        labels = assign_clusters(data, centroids)
        # 计算新的中心点
        new_centroids = update_centroids(data, labels, k)
        # 检查是否收敛
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# 运行K-Means算法
k = 3
centroids, labels = kmeans(data, k)

# 可视化结果
for i in range(k):
    plt.scatter(data[labels == i, 0], data[labels == i, 1])
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x')
plt.show()


# 2. 使用Scikit-learn库实现K-Means算法
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(42)
data = np.random.rand(100, 2)

# 定义K-Means模型并进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# 可视化结果
for i in range(3):
    plt.scatter(data[labels == i, 0], data[labels == i, 1])
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='x')
plt.show()

