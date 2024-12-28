from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2
#灰度化
img = cv2.imread("lenna.png")
h,w = img.shape[:2]
# img_gray = np.zeros([h,w], img.dtype)
# for i in range(h):
#     for j in range(w):
#         m = img[i, j]
#         img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
#
# print (img)
# print("image show gray: %s"%img_gray)
# cv2.imshow("image show gray",img_gray)
#
plt.subplot(221) #2行2列当前第1
img = plt.imread("lenna.png")
# img = cv2.imread("lenna.png", False)
plt.imshow(img)
print("---image lenna----")
print(img)
# 灰度化
# img_gray = rgb2gray(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(222)
plt.imshow(img_gray)
print("---image gray----")
print(img_gray)

img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary)
plt.show()
