import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png", 0)    # 以灰度图形式读取图像
print(img.shape)

rows, cols = img.shape

# 进行图像数据的格式处理
data = img.reshape((rows * cols), 1)
data = np.float32(data)

# 设定停止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# 使用cv2.kmeans()进行聚类—聚集成4类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)

# 生成最终图像—将图像像素还原成二维
dst = labels.reshape((img.shape[0], img.shape[1]))

plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
   plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]), plt.yticks([])
plt.show()
