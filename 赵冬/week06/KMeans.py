import random

import matplotlib.pyplot as plt
import numpy as np

def kmeans(X, k, eps, attempts):
    D = X
    labels = np.zeros((D.shape[0],))
    centers = D[random.sample(range(D.shape[0]), k)]
    # 尝试次数
    for i in range(attempts):
        center_change  = False
        # 计算每个点和中心点的距离
        for j in range(D.shape[0]):
            # 计算所有点到查询点的距离
            distances = np.sqrt(np.sum((D[j] - centers) ** 2, axis=1))
            # 找到距离最小的点的索引
            labels[j] = np.argmin(distances)
        # 中心计算中心点，如果中心点变化，则更新中心点
        for p in range(k):
            m = np.mean(D[labels == p], axis=0)
            # 如果新老质心的距离大于eps，则替换质心
            if np.linalg.norm(m - centers[p]) > eps:
                centers[p] = m
                center_change = True
        if not center_change:
            break
    return (labels, centers)
if __name__ == '__main__':
    X = np.array([[0.0888, 0.5885],
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
         ])
    labels, centers = kmeans(X, 3, 0.0000000001, 100)
    x = [n[0] for n in X]
    y = [n[1] for n in X]
    plt.scatter(x, y, c=labels, marker='o')

    x = [n[0] for n in centers]
    y = [n[1] for n in centers]
    plt.scatter(x, y, c=[0, 1, 2], marker='x')

    plt.show()
