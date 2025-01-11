from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 灰度化
img = cv2.imread("lenna.png")
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[2] * 0.11 + m[1] * 0.59 + m[0] * 0.3)

print(m)
print(img_gray)
print("image show gray: %s" % img_gray)
cv2.imshow("image show gray", img_gray)

plt.subplot(221)
img = plt.imread("lenna.png")

plt.imshow(img)
plt.title("Original Image")
print("---image lenna----")
print(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
plt.title("Gray Image ")

#二分化
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.title("Binary Image")
plt.show()
