from skimage.color import rgb2gray
import numpy as np
import  matplotlib.pyplot as plt
import cv2

img = cv2.imread("lenna.png")
a, b = img.shape[:2]
img_gray = np.zeros([a,b], img.dtype)
for i in range(a):
    for j in range(b):
        c = img[i, j]
        img_gray[i, j] = int(c[0]*0.11+c[1]*0.59+c[2]*0.3)
cv2.imshow("this is", img_gray)
# cv2.waitKey(0)

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

plt.subplot(222)
img_gray = rgb2gray(img)
plt.imshow(img_gray, cmap='gray')

plt.subplot(223)
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.imshow(img_binary, cmap='gray')

plt.show()
