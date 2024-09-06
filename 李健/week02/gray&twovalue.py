# -*- coding: utf-8 -*-
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

plt.subplot(221)
img = plt.imread("./origin_images/beautiful_girl01.jpg")
plt.imshow(img)
print("---image lenna----")
print(img)

# 灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

# 二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
