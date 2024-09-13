"""
@date 2024/09/07

@author 焦贤亮
@brief 彩色图像的灰度化、二值化
"""

import cv2
# import...as... 导入一个模块并为该模块指定一个别名
import numpy as np
import matplotlib.pyplot as plt
# import 用于导入整个模块，需要用模块名来调用函数
# from...import... 用于导入特定内容（函数、变量或类），无需使用模块名来调用
from PIL import Image
from skimage.color import rgb2gray


# 灰度化
img = cv2.imread("lenna.png")
# cv2.imread()读取图片后以多维数组的形式保存图片信息，前两维表示图片的像素坐标，最后一维表示图片的通道索引
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        pixel = img[i][j]
        # openCV读取的图片信息默认是BGR
        img_gray[i][j] = int(pixel[0]*0.11 + pixel[1]*0.59 + pixel[2]*0.3)

cv2.imshow("img_gray", img_gray)

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

# 灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap="gray")

# 二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(223)
plt.imshow(img_binary, cmap="gray")

plt.show()
