#coding:utf-8
"""
yzc- KMeans
1.获取原始灰度图
2.图像降维至一维
3.K-Means聚类
    设置停止条件criteria
    设置标签 flags
    cv2.kmeans
4.生成最终图像reshape，并imshow展示
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("..\\lenna.png",0)

data = np.float32(img.reshape((-1,1)))
print(data.shape)

criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
compactness,lables,centers = cv2.kmeans(data,4,None,criteria,10,flags)
print(compactness,lables,centers)

dst = lables.reshape((img.shape[0],img.shape[1]))
plt.rcParams['font.sans-serif']=['SimHei']
titles = ['原始图像','聚类图像']
images = [img,dst]
for i in range(len(images)):
    plt.subplot(1,2,i+1)
    plt.imshow(images[i],cmap='gray')
    plt.title(titles[i])
    plt.xticks([]);plt.yticks([])
plt.show()

