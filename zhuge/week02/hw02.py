import numpy as np
import cv2
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from PIL import Image

img = cv2.imread('lenna.png')
h,w = img.shape[:2]
print(h)
print(w)

#灰度化
img_gray = np.zeros((h,w),img.dtype)
for i in range(h):
    for j in range(w):
        p = img[i][j]
        img_gray[i][j] = int(p[0]*0.11 + p[1]*0.59 + p[2]*0.3)
# print(img_gray)
print("image show gray: %s"%img_gray)

cv2.imshow("image show gray: ",img_gray)

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("****image show*****")
print(img)

plt.subplot(222)
plt.imshow(img_gray,cmap='gray')
print("****image gray****")
print(img_gray)

#二值化
# Remove the alpha channel
img_rgb = img[..., :3]

# Convert to grayscale
img_gray = rgb2gray(img_rgb)
# img_gray = rgb2gray(img)

# for i in range(len(img_binary)):
#     img_binary[i] = [0 if j < 0.5 else 1 for j in img_binary[i]]
img_binary = np.where(img_gray >= 0.5, 1, 0)

# print("****img_binary show *****", img_binary)

plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
plt.show()
