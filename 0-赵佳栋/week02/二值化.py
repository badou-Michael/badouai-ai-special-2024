'''
@Project ：BadouCV 
@File    ：test03.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/10/10 15:26 
'''
import numpy as np
import matplotlib.pyplot as plt
import cv2


# 读取RGB的灰度图
img = cv2.imread("../lenna.png", cv2.IMREAD_GRAYSCALE)

# 遍历每个像素，进行二值化
h, w = img.shape
threshold=127 # 0-255范围的中间阈值
binary_img = np.zeros_like(img)
for i in range(h):
    for j in range(w):
        if img[i, j] >= threshold:
            binary_img[i, j] = 255
        else:
            binary_img[i, j] = 0
plt.subplot(121)
plt.imshow(img, cmap='gray')

plt.subplot(122)
plt.imshow(binary_img, cmap='gray')
plt.show()
