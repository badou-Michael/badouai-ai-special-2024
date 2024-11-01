import numpy as np
import matplotlib.pyplot as plt


# K-means 算法实现
def kmeans(X, k, max_iters=100):
    # 随机选择初始聚类中心
    initial_centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    centroids = initial_centroids
    for _ in range(max_iters):
        # 计算距离并分配类别
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # 更新聚类中心
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # 检查收敛
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return labels, centroids


# 示例数据点
data = np.array([[5, 2], [2, 3], [1, 0],
                 [4, 1], [4, 2], [4, 0]])

# 聚类数量
k = 2

# 运行 K-means
labels, centroids = kmeans(data, k)

# 可视化结果
colors = ['r', 'g']
plt.figure(figsize=(8, 6))
for i in range(k):
    plt.scatter(data[labels == i, 0], data[labels == i, 1], color=colors[i], label=f'Cluster {i + 1}')
plt.scatter(centroids[:, 0], centroids[:, 1], color='b', marker='X', s=200, label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid()
plt.show()
