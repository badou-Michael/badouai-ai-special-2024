import cv2
import numpy as np

img = cv2.imread('photo1.jpg')
result3 = img.copy()

# src和dst的输入是图像对应的顶点坐标
src = np.float32([[207,151],[517,285],[17,601],[343,731]])
dst = np.float32([[0,0],[337,0],[0,488],[337,488]])
print(img.shape)

# 生成透视变换矩阵，进行透视变换
m = cv2.getPerspectiveTransform(src,dst)
print('warpMatrix:')
print(m)
result = cv2.warpPerspective(result3, m, (337,488))
cv2.imshow('src',img)
cv2.imshow('resulr',result)
cv2.waitKey(0)

#K-Means
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取原始图像灰度颜色
img = cv2.imread('lenna.png',0)
print(img.shape)

# 获取高度、宽度
rows,cols = img.shape[:]

# 图像二维像素转换成一维
data = img.reshape((rows * cols, 1))
data = np.float32(data)

# 停止条件(type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means聚类，聚成4类
compactness, lables, centers = cv2.kmeans(data, 4, None,criteria, 10,flags)

# 生成最终图像
dst = lables.reshape((img.shape[0],img.shape[1]))

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
titles = [u'原始图像',u'聚类图像']
images = [img,dst]
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray'),
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    plt.show()
