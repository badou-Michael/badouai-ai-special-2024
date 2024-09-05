# 作业：实现彩色图片的灰度化和二值化

# 导包
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
from PIL import Image

# 灰度化
# 获取图片
img = cv2.imread("lenna.png")
# 获取图片的high和width
h,w = img.shape[:2] # 512 512
# 创建和当前图像大小相等的单通道图片
img_gray = np.zeros([h,w],img.dtype)
# 遍历图像
for i in range(h):
    for j in range(w):
        # 取出当前high和width中的BGR坐标
        m = img[i,j]
        # 将BGR坐标转化为gray坐标并赋值给新图像(浮点算法)
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
print(m) # 打印最后一个像素的BGR值
print(img_gray) # 打印灰度化后的图像矩阵
print("image show gray: %s"%img_gray) # 打印灰度图像矩阵的字符串表示
cv2.imshow("image show gray",img_gray) # 使用OpenCV显示灰度图像

# 使用matplotlib显示原图像
plt.subplot(221) # 创建一个2*2的子图，当前是第一个
img = plt.imread("lenna.png") # 使用matplotlib读取图像（再次读取）
plt.imshow(img) # 显示原图像
print("---image lenna----")
print(img) # 打印原图像的数组表示

# 灰度化
img_gray = rgb2gray(img) # 将RGB图像转换为灰度图像
plt.subplot(222) # 创建一个2*2的子图，当前是第2个
plt.imshow(img_gray, cmap='gray') # 系那是灰度图像，指定颜色映射为gray
print("---image gray----")
print(img_gray) # 打印灰度图像的数组表示

# 二值化
# 如果灰度值大于等于0.5，设为1，反之设为0
img_binary = np.where(img_gray >= 0.5, 1, 0)
# 创建一个2*2的子图，当前是第3个
plt.subplot(223)
# 显示二值化图像
plt.imshow(img_binary, cmap='gray')

# 显示所有子图
plt.show()
