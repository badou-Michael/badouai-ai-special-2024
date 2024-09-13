import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

#读取图片
image=cv2.imread('lenna.png')

h, w = image.shape[:2] #获取图像矩阵前两个参数，即高和宽，第三个参数为通道
image_grey=np.zeros([h,w],image.dtype) #创建一个与image相同大小的0矩阵

#灰度化 方法一
for i in range(h):
    for j in range(w):
        m = image[i,j]
        image_grey[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

#显示灰度化后的照片
#cv2.imshow('image_grey',image_grey)
#cv2.waitKey()
plt.subplot(221)
plt.imshow(image_grey, cmap='gray')

#灰度法 方法二
image_grey2 = rgb2gray(image)
plt.subplot(222)
plt.imshow(image_grey2, cmap='gray')

#二值化

img_binary = np.where(image_grey/255 >= 0.5, 1, 0)

plt.subplot(223)
plt.imshow(img_binary , cmap='gray')
plt.show()

