from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 作业1
img = cv2.imread('D:/Users/Admin_FCL/Desktop/opencv/lenna.png')
h, w = img.shape[:2]
img_gray = np.zeros([h,w], img.dtype) #创建一个与图大小一样的零矩阵
for i in range(h):
    for j in range(w):
        m = img[i,j] 
        img_gray [i,j]= int(m[0]*0.11+m[1]*0.59+m[2]*0.3) #BRG


cv2.imshow('Grayscale Image', img_gray) # 直接以灰度图的形式打开
cv2.waitKey(0)
cv2.destroyAllWindows()


#或者直接调用内置函数 
img_gray = rgb2gray(img)
plt.imshow(img_gray, cmap='gray')
plt.axis('off')
plt.show()

# 作业2
#二值化，黑白图
img_gray = img_gray/255
print (img_gray)

img_binary = np.where(img_gray >= 0.5, 1, 0)
print(img_binary)
