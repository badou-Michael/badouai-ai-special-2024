# 导入常用第三方模块
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# # 法一
# # 读取原图像lenna.png
# image = cv.imread('E:\GUO_APP\GUO_AI\picture\lenna.png')
# image1 = cv.cvtColor(image,cv.COLOR_BGR2RGB)               # BGR 转为 RGB
# plt.subplot(221)                                            # 建立 2*2的画布
# plt.imshow(image1)                                            # 创建并配置图像
# plt.show()                                                     # 展示图像image1
#
# # 灰度化
# image = cv.imread('E:\GUO_APP\GUO_AI\picture\lenna.png',0)
# cv.imwrite('E:\GUO_APP\GUO_AI\picture\lenna_gray.png',image)
# image2 = plt.imread('E:\GUO_APP\GUO_AI\picture\lenna_gray.png')
# #cv.imshow('test',image2)
# plt.subplot(222)
# plt.imshow(image2,cmap='gray')
# plt.show()
#
# # 二值化
# image3 = np.where(image2 >= 0.5,1,0)
# plt.subplot(223)
# plt.imshow(image3,cmap='gray')
# plt.show()
# # 图像等待-直到按任意键消失
# cv.waitKey(0)
# 法二
# 读取原图lenna 并展示灰度化的lenna_gray,主要是利用np.zeros 创建空值数组，将BGR图的每个像素的坐标二值化，展示单通道灰度图
image = cv.imread('E:/GUO_APP/GUO_AI/picture/lenna.png')
h,w = image.shape[0:2]
image1 = np.zeros([h,w],image.dtype)
for i in range(h):
    for j in range(w):
         m = image[i,j]
         image1[i,j] = int((m[0]*0.11 + m[1]*0.59 + m[2]*0.3))
cv.imshow('lenna1',image)
cv.imshow('lenna1_gray',image1)

# 二值化
image2 = np.zeros([h,w],image.dtype)
for i in range(h):
    for j in range(w):
        if image1[i,j] <= 123:
            image2[i,j]= 0
        else:
            image2[i,j]= 1
plt.imshow(image2,cmap='gray')
plt.show()
cv.waitKey(0)


