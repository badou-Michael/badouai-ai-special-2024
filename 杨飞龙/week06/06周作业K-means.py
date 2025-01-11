import numpy as np
import cv2 as cv

# # 实现Kmeans彩色图（用center显示）
# img = cv.imread('lenna.png')
# # 将多维数组img转化为一个二维数组
# Z = img.reshape((-1,3))
# # 转换为 np.float32
# Z = np.float32(Z)
# # 定义标准、簇数（K）并应用 kmeans()
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 4
# ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# # 现在转换回 uint8，并生成原始图像
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((img.shape))
# cv.imshow('res2',res2)
# cv.waitKey(0)

# # 实现Kmeans灰度图（用center显示）
# img = cv.imread('lenna.png',0)
# # 将多维数组img转化为一个二维数组
# Z = img.reshape((-1,1))
# # 转换为 np.float32
# Z = np.float32(Z)
# # 定义标准、簇数（K）并应用 kmeans()
# criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)
# K = 4
# ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# # 现在转换回 uint8，并生成原始图像
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((img.shape))
# cv.imshow('res2',res2)
# cv.waitKey(0)

# 实现Kmeans灰度图(用labels显示)
img = cv.imread('lenna.png',0)
# 将多维数组img转化为一个二维数组
Z = img.reshape(-1,1)
# 转换为 np.float32
Z = np.float32(Z)
# 定义标准、簇数（K）并应用 kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)
K = 4
ret,labels,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# 现在转换回 uint8，并生成原始图像
print("labels",labels)
print("center",center)
img_mean = np.uint8(labels * 255 / (K - 1))
print("labels_change",img_mean)
# 将一维的labels数组重塑回图像的形状
img_mean = img_mean.reshape(img.shape)
cv.imshow('res2',img_mean)
cv.waitKey(0)
