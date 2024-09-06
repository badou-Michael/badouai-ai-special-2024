import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray


img = cv2.imread("lenna.png")                #读取图像
h,w = img.shape[:2]                          #获取图片的高和宽，img.shape获取图片的形状，
                                             # 图片是彩色，返回一个包含行数（高度）、列数（宽度）和通道数的元组，灰度图只返回行数和列数
img_gray = np.zeros([h,w],img.dtype)            #np.zeros 快速创建指定形状和类型且所有元素初始化为零的数组
                                                #img.dtype 获取图像的数据类型


for i in range(h):
    for j in range(w):
        m = img[i,j]                           #获取当前每个像素的BGR坐标
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)   #灰度化处理

print(m)
print(img_gray)
print("image show gray:%s" %img_gray)
cv2.imshow("image show gray",img_gray)     #cv2.imshow  用于指定窗口中显示 OpenCV 图像，窗口自适应图像大小



plt.subplot(221)                            #画布 221表示将画布分为2行2列，1表示第一行第一列的画布分区
img = plt.imread("lenna.png")
plt.imshow(img)
print("lenna picture")
print(img)

img_gray = rgb2gray(img)                     #转化为灰度图
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')            # cmap参数用于指定色彩映射,决定了图像的颜色显示方式。
print("lenna gray picture")
print(img_gray)



img_binary = np.where(img_gray >= 0.5, 1, 0)
print("lenna binary picture")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
plt.show()




