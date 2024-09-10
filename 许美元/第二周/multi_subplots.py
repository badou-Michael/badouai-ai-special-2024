
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

# 设置全局字体
matplotlib.rcParams['font.family'] = 'SimHei'  # 设置字体名称
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示符号


ori_img = plt.imread("lena.png")
plt.subplot(331)
plt.imshow(ori_img)
plt.title('原图')


### 作业一：实现灰度化
## 方法1：使用opencv和numpy根据灰度化公式进行换算
img = cv2.imread("lena.png")
h, w = img.shape[:2]  # 获取数组（图像）的高度（行数）和宽度（列数）。
img_gray = np.zeros(shape=[h, w], dtype=img.dtype)  # 创建一个与原始图像具有相同尺寸和数据类型的全零数组
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 获取位于第 i 行、第 j 列的像素值。
        # print(m)  # 在RGB图像中，m 是一个包含三个元素的数组（或列表），分别对应于该像素的红色、绿色和蓝色通道的强度值。
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像

plt.subplot(334)
plt.imshow(img_gray, cmap='gray')
plt.title('使用opencv和numpy\n根据灰度化公式进行换算')


## 方法2：使用opencv的属性
img_gray_2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.subplot(335)
plt.imshow(img_gray_2, cmap='gray')
plt.title('使用opencv的属性')


# 方法3：使用skimage
img_gray_3 = rgb2gray(img)

plt.subplot(336)
plt.imshow(img_gray_3, cmap='gray')
plt.title('使用skimage的属性')


### 作业二：实现二值化
img_gray = rgb2gray(img)
img_binary = np.where(img_gray >= 0.5, 1, 0)

plt.subplot(337)
plt.imshow(img_binary, cmap='gray')
plt.title('实现二值化')

# 调整子图间距
plt.tight_layout()
# 显示图片
plt.show()
