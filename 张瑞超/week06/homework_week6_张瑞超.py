import numpy as np
import matplotlib.pyplot as plt
import cv2


def kmeans(data, K, criteria, attempts=10, flags='random', initial_centers=None):
    """
    参数:
    data: numpy.ndarray, 输入的数据集，最好是np.float32类型
    K: int, 期望的聚类簇数
    criteria: tuple, 迭代终止的标准，格式为 (type, max_iter, epsilon)
    attempts: int, 尝试KMeans的次数，算法返回最优结果
    flags: str, 初始质心的选择方式 ('random' 表示随机选择初始中心, 'pp' 表示使用KMeans++)
    initial_centers: numpy.ndarray, 可选，初始化质心，如果不传则根据flags选择方式

    返回:
    best_error: float, 最优聚类的总误差（点到质心的距离平方和）
    best_labels: numpy.ndarray, 每个数据点的聚类标签
    best_centers: numpy.ndarray, 每个簇的质心
    """
    def calculate_distance(points, centers):
        # 使用广播机制和矢量化操作来计算所有点到所有质心的距离
        return np.linalg.norm(points[:, np.newaxis] - centers, axis=2)

    def update_centers(data, labels, K):
        # 计算每个簇的均值来更新质心，利用NumPy的高效分组运算
        new_centers = np.zeros((K, data.shape[1]), dtype=np.float32)
        for i in range(K):
            points_in_cluster = data[labels == i]
            if len(points_in_cluster) > 0:
                new_centers[i] = np.mean(points_in_cluster, axis=0)
            else:
                # 如果某簇为空，随机选择一个点作为新的质心
                new_centers[i] = data[np.random.choice(data.shape[0])]
        return new_centers

    def kmeans_pp_init(data, K):
        """实现 KMeans++ 初始化质心选择算法"""
        centers = []
        # 随机选择第一个质心
        centers.append(data[np.random.choice(data.shape[0])])

        # 选择接下来的K-1个质心
        for _ in range(1, K):
            # 计算每个点到最近质心的最小距离平方
            distances = np.min(calculate_distance(data, np.array(centers)), axis=1)
            distances_squared = distances ** 2

            # 根据距离平方选择下一个质心（概率分布）
            probs = distances_squared / np.sum(distances_squared)
            cumulative_probs = np.cumsum(probs)
            r = np.random.rand()

            # 选择满足随机数的下一个质心
            next_center_idx = np.where(cumulative_probs >= r)[0][0]
            centers.append(data[next_center_idx])

        return np.array(centers)

    best_error = float('inf')
    best_labels = None
    best_centers = None

    for attempt in range(attempts):
        # 初始化聚类中心
        if flags == 'random':
            if initial_centers is None:
                centers = data[np.random.choice(data.shape[0], K, replace=False)]
            else:
                centers = initial_centers.copy()
        elif flags == 'pp':
            centers = kmeans_pp_init(data, K)

        labels = np.zeros(data.shape[0], dtype=np.int32)

        for iteration in range(criteria[1]):
            # 计算每个点到每个质心的距离，分配到最近的簇，使用向量化操作
            distances = calculate_distance(data, centers)
            labels = np.argmin(distances, axis=1)

            # 更新质心，利用NumPy分组均值
            new_centers = update_centers(data, labels, K)

            # 终止条件
            if criteria[0] & 1:  # TERM_CRITERIA_EPS
                if np.linalg.norm(new_centers - centers) < criteria[2]:
                    break
            if criteria[0] & 2:  # TERM_CRITERIA_MAX_ITER
                if iteration >= criteria[1]:
                    break

            centers = new_centers

        # 计算误差（距离的平方和）
        error = np.sum(np.min(calculate_distance(data, centers), axis=1) ** 2)

        # 找到最佳结果
        if error < best_error:
            best_error = error
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_error, best_labels, best_centers



# 读取原始图像为灰度图
img = cv2.imread('lenna.png', 0)
print(img.shape)

# 获取图像高度、宽度
rows, cols = img.shape[:]

# 将二维像素转换为一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)

# 停止条件 (type, max_iter, epsilon)
criteria = (3, 10, 1.0)

# 设置标签（标志为随机选择初始中心）
flags = 'pp'

compactness, labels, centers = kmeans(data, 4, criteria, 10, flags)

# 生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
