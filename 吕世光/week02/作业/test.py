# 引入灰度处理
from skimage.color import rgb2gray
# 引入numpy库处理数组，矩阵
import numpy as np
# 引入绘图库
import matplotlib.pyplot as plt
# 引入opencv
import cv2

img = plt.imread("lenna.png")
# 创建图表 将原图画于221位置
plt.subplot(221)
plt.imshow(img)
# 第一种方法，遍历灰度--------------->
# 读取图片
img = cv2.imread("lenna.png")
# 取出图像的宽，高，即分辨率
h, w = img.shape[:2]
# 创建空白图片画布数组
# img.dtype 依据img的数据类型创造
img_gray = np.zeros((w, h), img.dtype)
# 遍历512*512 得到灰度(通过浮点方法)，设置空白画布的矩阵数据
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 原图每个点的BGR数据
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
# 第二种方法，skimage插件灰度--------------->
# img_gray = rgb2gray(img)
# 第三种方法，cv插件灰度--------------->
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

# 二值化
# 第一种方法，遍历二阶--------------->
h1, w1 = img_gray.shape[:2]
for l in range(h1):
    for m in range(w1):
        if img_gray[l, m] / 255 < 0.5:
            img_gray[l, m] = 0
        else:
            img_gray[l, m] = 1
# 第二种方法，np处理数据--------------->
# img_gray = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(223)
plt.imshow(img_gray, cmap='gray')
plt.show()
