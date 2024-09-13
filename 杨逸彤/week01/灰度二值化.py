from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread("d:\\Users\ls008\Desktop\lenna.png")
print(img)
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
#灰度化
#原理做法：
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
#函数做法
img_gray_f = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(img_gray)

#二值化
#原理做法：
h_gray,w_gray = img_gray.shape[:2]
img_binary = np.zeros([h_gray,w_gray],img_gray.dtype)
for i in range(h_gray):
    for j in range(w_gray):
        g = img_gray[i,j]
        if(g/255 <= 0.5):
            img_binary[i,j] = 0
        else:
            img_binary[i, j] = 1
#函数做法：
img_binary_f = np.where(img_gray/255 >= 0.5, 1, 0)

plt.subplot(221)
plt.imshow(img)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')

plt.show()










