import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('photo1.jpg', 0)
print(img.shape)
rows, cols = img.shape[:]

#二维转换为一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)

#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类
compactness, labels, centers = cv2.kmeans(data, 5, None, criteria, 12, flags)

#生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))

#正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'),
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()
