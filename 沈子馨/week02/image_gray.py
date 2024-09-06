"""
第二周作业：实现彩色图像的灰度化和二值化
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import cv2

#彩色图像灰度化
#cv2方法
img = cv2.imread("lenna.png")  #读图
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #opencv中读入的BGR转为RGB
h, w = img.shape[:2]
img_gray = np.zeros([h,w], img.dtype)
for i in range(h):
     for j in range(w):
         m = img[i,j]
         img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
print (m)
print (img_gray)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray", img_gray)
cv2.waitKey(5)

#plt方式
plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("---image lenna----")
print(img)

#灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

#二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()

