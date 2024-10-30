import numpy as np
import cv2

def initialize_centers(data, k):
    """
    随机初始化 k 个聚类中心
    :param data: 数据集 (N, D) 形状的数组，N 是数据点数量，D 是特征维度
    :param k: 聚类数量
    :return: 初始化的聚类中心 (k, D) 形状的数组
    """
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_clusters(data, centers):
    """
    将每个数据点分配到最近的聚类中心
    :param data: 数据集 (N, D) 形状的数组
    :param centers: 聚类中心 (k, D) 形状的数组
    :return: 分配的聚类标签 (N,) 形状的数组
    """
    distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
    return np.argmin(distances, axis=1)

def update_centers(data, labels, k):
    """
    更新聚类中心
    :param data: 数据集 (N, D) 形状的数组
    :param labels: 分配的聚类标签 (N,) 形状的数组
    :param k: 聚类数量
    :return: 新的聚类中心 (k, D) 形状的数组
    """
    new_centers = []
    for i in range(k):
        cluster_data = data[labels == i]
        if len(cluster_data) > 0:
            new_center = np.mean(cluster_data, axis=0)
        else:
            new_center = data[np.random.choice(data.shape[0])]
        new_centers.append(new_center)
    return np.array(new_centers)

def kmeans(data, k, max_iters=100, tol=1e-4):
    """
    K-Means 聚类算法
    :param data: 数据集 (N, D) 形状的数组
    :param k: 聚类数量
    :param max_iters: 最大迭代次数
    :param tol: 收敛阈值
    :return: 聚类中心 (k, D) 形状的数组，分配的聚类标签 (N,) 形状的数组
    """
    centers = initialize_centers(data, k)
    for _ in range(max_iters):
        labels = assign_clusters(data, centers)
        new_centers = update_centers(data, labels, k)
        if np.linalg.norm(new_centers - centers) < tol:
            break
        centers = new_centers
    return centers, labels

def apply_kmeans_to_image(image, k):
    """
    将 K-Means 聚类应用于图像
    :param image: 输入图像 (H, W, C) 形状的数组
    :param k: 聚类数量
    :return: 聚类后的图像 (H, W, C) 形状的数组
    """
    height, width, channels = image.shape
    data = image.reshape(-1, channels).astype(np.float32)
    centers, labels = kmeans(data, k)
    new_image = centers[labels].reshape(height, width, channels).astype(np.uint8)
    return new_image

# 读取图像
image = cv2.imread("lenna.png")

# 定义不同的聚类数量
k_values = [5, 10, 15, 20]

# 存储聚类后的图像
clustered_images = []

for k in k_values:
    new_image = apply_kmeans_to_image(image, k)
    clustered_images.append(new_image)

# 拼接图像
height, width, _ = image.shape
result_image = np.hstack(clustered_images)

# 显示结果
cv2.imshow("Original Image", image)
cv2.imshow("Clustered Images (5, 10, 15, 20)", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果图像
cv2.imwrite("clustered_images.png", result_image)
