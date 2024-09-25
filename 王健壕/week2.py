# -*- coding: utf-8 -*-
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread("lenna.png")
# 灰度化
img_gray = rgb2gray(img)
plt.subplot(221)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")

plt.subplot(222)
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)

plt.imshow(img_binary, cmap='gray')
plt.show()
