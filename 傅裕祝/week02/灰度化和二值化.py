import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2

# 灰度化
# 一
img=plt.imread('lenna.png')
img_gray=rgb2gray(img)
# 二
# img=cv2.imread('lenna.png')
# h,w = img.shape[:2]
# img_gray = np.zeros([h,w],img.dtype)
# for i in range(h):
#     for j in range(w):
#         m = img[i,j]
#         img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
# cv2.imshow("image show gray",img_gray)
# 输出原图
plt.subplot(221)
plt.imshow(img)

# 输出灰度化图
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')

# 二值化
# 一
rows,cols=img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i,j]>=0.5:
            img_gray[i,j]=1
        else:img_gray[i,j]=0
plt.subplot(223)
plt.imshow(img_gray,cmap='gray')

#二
# img_binary=np.where(img_gray>=0.5,1,0)
# plt.subplot(224)
# plt.imshow(img_binary,cmap='gray')
# print(img_binary)
plt.show()










