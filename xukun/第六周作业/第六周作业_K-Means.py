import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def k_means_clustering(img, k):
    # 1. 读取图像
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

    # 2. 重塑图像数据
    pixels = image.reshape(-1, 3)  # 将图像重塑为二维数组，每行是一个像素的 RGB 值
    pixels = np.float32(pixels)  # 将数据类型转换为 float32
    # 3. 应用 K-means 聚类算法
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #  4. 显示聚类中心
    compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    #  5. 显示聚类结果
    KMeans_img = labels.reshape(image.shape[0], image.shape[1])

    # 6. 显示原图和聚类后的图像
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title(f'K-means(k={k})')
    plt.imshow(KMeans_img)

    plt.show()


if __name__ == '__main__':
    k_means_clustering('lenna.png', 3)
