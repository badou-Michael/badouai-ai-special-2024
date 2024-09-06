from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 原图
img = cv2.imread("lenna.png")
print("-----img lenna shape-----")
print(img.shape)
# print("-----img lenna-----")
# print(img)
plt.subplot(131)
plt.imshow(img)
plt.title("img")

# 灰度化--法一
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
print("-----img_gray shape------")
print(img_gray.shape)
# print("-----img_gray------")
# print(img_gray)
plt.subplot(132)
plt.imshow(img_gray, cmap='gray')
plt.title("img_gray")

# 二值化--法二
img_binary = np.where(img_gray >= 0.5*255, 255, 0)
print("-----img_binary shape------")
print(img_binary.shape)
# print("-----img_binary------")
# print(img_binary)
plt.subplot(133)
plt.imshow(img_binary, cmap='gray')
plt.title("img_binary")

plt.show()
