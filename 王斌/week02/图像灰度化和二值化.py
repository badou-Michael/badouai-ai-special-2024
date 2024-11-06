from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

import cv2



img = cv2.imread("C:/Users/Administrator/Desktop/123.jpg")
h,w = img.shape[:2]
img_gray = np.zeros([h,w],dtype=float)
print(img_gray)
#灰度化
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = (m[0]*0.11+m[1]*0.59+m[2]*0.3)/255
print(img_gray)
plt.subplot(221)
plt.imshow(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
#二值化
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i, j] <= 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1;
plt.subplot(223)
plt.imshow(img_gray, cmap='gray')



img = plt.imread("C:/Users/Administrator/Desktop/123.jpg")

#灰度化   归一化
img_gray = rgb2gray(img)

#灰度化
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)

print(img_gray)
# plt.subplot(224)
# plt.imshow(img_gray, cmap="gray")
plt.show()
