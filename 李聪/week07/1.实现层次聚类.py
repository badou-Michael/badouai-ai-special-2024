import cv2
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 读取图像并进行灰度转换
img1 = cv2.imread('iphone1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('iphone2.png', cv2.IMREAD_GRAYSCALE)

# 创建SIFT对象并提取特征点和描述符
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# 将两个图像的描述符结合用于层次聚类
descriptors = np.vstack((descriptors1, descriptors2))

# 层次聚类并绘制树状图
linked = linkage(descriptors, method='ward')
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title("Dendrogram for SIFT Feature Clustering")
plt.xlabel("Feature Index")
plt.ylabel("Euclidean Distance")
plt.show()
