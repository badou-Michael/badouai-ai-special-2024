# -*- coding: utf-8 -*-
"""
@Author: zl.chen
@Name: img2gray.py
@Time: 2024/9/5 10:05
@Desc:  图片灰度化
"""
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np

img = plt.imread("./lenna.png")
plt.subplot(221)
plt.imshow(img)
plt.show()

# 灰度化
gray_img = rgb2gray(img)
plt.subplot(222)
plt.imshow(gray_img)
plt.show()

# 二值化
binary_img = np.where(gray_img >= 0.5, 1, 0)
plt.subplot(223)
plt.imshow(binary_img)
plt.show()
