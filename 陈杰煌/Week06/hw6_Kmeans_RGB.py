import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像
img = cv2.imread('lenna.png')
print(img.shape)

# 图像转换为一维数据并转换类型
data = img.reshape((-1, 3)).astype(np.float32)

# 定义停止条件和标签
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

# 定义聚类数目列表
K_values = [2, 4, 8, 16, 64]
results = []

# 对不同的K值进行聚类
for K in K_values:
    _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, flags)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    dst = res.reshape(img.shape)
    results.append(dst)

# 图像转换为RGB格式以供显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = [cv2.cvtColor(dst, cv2.COLOR_BGR2RGB) for dst in results]

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = ['原始图像'] + [f'聚类图像 K={K}' for K in K_values]
images = [img] + results
for i in range(len(images)):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
