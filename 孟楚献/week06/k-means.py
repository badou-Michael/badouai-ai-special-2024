import random

import numpy as np

# 计算两个二维坐标的长度
def length(point1, point2):
    return pow((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2, 0.5)

class KMeans:
    def __init__(self, X, k, max_iter=10000):
        self.X = X
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.pridict_res = None

    # 随机初始化质心
    def _init_centroid(self):
        indices = random.sample(range(len(self.X)), self.k)
        self.centroids = np.array(self.X[indices, :])
        print(self.centroids)

    # 数据点分配到聚类中心
    def _allocate_clusters(self):
        clusters = {}
        self.pridict_res = []
        for _ in range(self.k):
            clusters[_] = []
        for point in self.X:
            distance = [np.linalg.norm(point - centroid) for centroid in self.centroids]
            cluster_index = distance.index(min(distance))
            clusters[cluster_index].append(point)
            self.pridict_res.append(cluster_index)
        print(clusters)
        return clusters

    # 更新质心
    def _update_centroid(self, clusters):
        new_centroids = []
        for cluster_index, points in clusters.items():
            new_centroid = np.mean(points, axis=0)
            new_centroids.append(new_centroid)
        return np.array(new_centroids)

    # 执行KMeans算法
    def fit(self):
        self._init_centroid()
        for _ in range(self.max_iter):
            clusters = self._allocate_clusters()
            new_centroids = self._update_centroid(clusters)
            if (new_centroids == self.centroids).all():
                break
            self.centroids = new_centroids
        return self.centroids, clusters

    def pridict(self):
        self.fit()
        return self.pridict_res



X = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]
    ]
X = np.mat(X)
print(len(X))
km = KMeans(X, 3)
print(km.fit())
print(km.pridict())