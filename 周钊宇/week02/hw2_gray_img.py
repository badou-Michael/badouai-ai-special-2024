"""
实现灰度化，二值化
"""

import matplotlib.pyplot as plt
import cv2



#实现灰度化

# img = cv2.imread("lenna.png")
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = plt.imread("lenna.png")
plt.figure()
plt.subplot(221)
plt.imshow(img)

# gray_img = rgb2gray(img)
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.subplot(222)
plt.imshow(gray_img, cmap="gray")
# print(gray_img)
# print(gray_img.shape)

#实现二值化
_, binary_img = cv2.threshold(gray_img, 127/255, 1, cv2.THRESH_BINARY)
plt.subplot(223)
plt.imshow(binary_img, cmap="gray")
# print(binary_img)
# print(binary_img.shape)


plt.show()


