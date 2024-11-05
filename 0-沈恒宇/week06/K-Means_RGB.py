# coding: utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt


def kmeans_clustering(data, k, criteria, flags):
    """执行K-Means聚类并返回结果"""
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    """labels.flatten()将多维数组labels转换为一维数组。
    例如，如果labels是一个二维数组，labels.flatten()会将其转换为一维数组。"""
    return res.reshape((img.shape))


def convert_to_rgb(image):
    """将BGR图像转换为RGB"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# 读取原始图像
img = cv2.imread('lenna.png')
if img is None:
    raise ValueError("Image not found")

print(img.shape)

# 图像二维像素转换为一维
data = img.reshape((-1, 3))
"""
这行代码将原始图像的二维数组（形状为（height，width，3））转换为一维数组（形状为（-1,3））
-1 表示自动计算这一维度的大小，以确保总元素数量不变
3 是因为图像的每个像素通常有三个颜色通道（红、绿、蓝），所以每个像素点可以用一个包含三个元素的向量来表示。
"""
data = np.float32(data)

# 停止条件 (type, max_iter, epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K值列表
k_list = [2, 4, 8, 16, 64]

# 执行K-Means聚类并存储结果
results = {}
for k in k_list:
    results[f'K={k}'] = kmeans_clustering(data, k, criteria, flags)

# 图像转换为RGB显示
img_rgb = convert_to_rgb(img)
results['原始图像'] = img_rgb

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = [f'聚类图像 K={k}' for k in k_list] + ['原始图像']
images = list(results.values())

for i in range(len(images)):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()