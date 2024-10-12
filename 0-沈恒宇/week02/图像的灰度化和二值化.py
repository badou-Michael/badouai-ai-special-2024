import cv2
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('lenna.png')

# 将BGR图像转化为RGB类型
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 灰度化
img_gray = rgb2gray(img_rgb)

# 二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)

# 2*2方格，第1个放原图
plt.subplot(221)
plt.imshow(img_rgb)

# 第2个放灰度化的图像
plt.subplot(222)
plt.imshow(img_gray, cmap="gray")  # 使用camp = “gary” ，使图像以灰度模式显示

# 第3个放二值化图像
plt.subplot(223)
plt.imshow(img_binary, cmap="gray")

# 显示图像
plt.show()
