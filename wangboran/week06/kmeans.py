# -*- coding: utf-8 -*-
# author: 王博然
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 以灰度方式读入
img = cv2.imread('../lenna.png', 0)
rows, cols = img.shape[:]

# 转化为1维
data = img.reshape((rows * cols, 1))
data = np.float32(data)

# 停止条件 精度+次数 混合
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, \
            10, 1)

K = 4
compactness, labels, centers = cv2.kmeans(       \
    data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 从1维再转为2维
dst = labels.reshape((rows, cols))

#显示图像
titles = ['original', 'kmeans']  
images = [img, dst]  
for i in range(2):  
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'), 
   plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  
plt.show()