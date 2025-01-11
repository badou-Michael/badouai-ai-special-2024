import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 读取示例图并转换为二维数组
image = cv2.imread('photo1.jpg')
data = image.reshape((-1, 3)).astype(np.float32)

# 使用肘部法则估计合适的K值
def estimate_k(data, max_k=10):
    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        distortions.append(kmeans.inertia_)

    # 绘制肘部图
    plt.plot(range(1, max_k + 1), distortions, marker='o')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal K')
    plt.show()

    # 手动选择一个 K 值
    return int(input("根据肘部图选择一个K值: "))

# 估计 K 值
K = estimate_k(data)

# 使用 K-Means 聚类
kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# 将聚类结果应用到图像
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(image.shape)

# 显示原始图像和聚类结果
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
