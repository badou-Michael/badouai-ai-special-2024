import cv2 as cv
import numpy as np


img = cv.imread('lenna.png',0)

rows, cols = img.shape[:]

data = img.reshape((rows * cols, 1))  #数据转换为一维
data = np.float32(data)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)  #迭代终止条件的设置

flags = cv.KMEANS_RANDOM_CENTERS  #标签的设置

compactness, labels, centers = cv.kmeans(data, 4, None, criteria, 10, flags)   #聚类

dst = labels.reshape((img.shape[0], img.shape[1]))


plt.figure()
plt.imshow(dst)
plt.gray()
plt.show()
