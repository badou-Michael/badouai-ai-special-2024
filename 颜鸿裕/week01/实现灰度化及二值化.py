from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread("lenna.png")
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 +m[2]*0.3)

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

img_binary = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img_gray[i,j]/255
        if m <= 0.5:
            img_binary[i,j] = 0
        else:
            img_binary[i,j] = 1
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
