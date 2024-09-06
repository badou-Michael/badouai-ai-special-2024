'''
1.实现RGB2GRAY（手工实现+调接口） 2.实现二值化
'''

import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from PIL import Image
# import cv2
# from skimage.color import rgb2gray

img = Image.open("lenna.png")       # 读取模式为RGB
print(img.size)     # (512,512)  获取图像的高度和宽度，返回值是元组tuple
print(img)
img_array = np.array(img)       # 将img转换为numpy数组，以便高效处理像素数据
print(img_array.shape)      # img_array是一个形状为（高度，宽度，颜色通道）的三维数组。
h, w, _ = img_array.shape       # 获取图片的高度和宽度
img_gray = np.zeros([h, w], img_array.dtype)        # 创建一个指定了形状和数据类型的全零数组
# print(img_gray)
for i in range(h):
    for j in range(w):
        m = img_array[i, j]
        img_gray[i, j] = int(0.3*m[0]+0.59*m[1]+0.11*m[2])
print("-----img_gray-----")
print(img_gray.shape)
print(img_gray)


# 调用PIL中的convert接口实现RGB2GRAY
auto_img_gray = img.convert('L')        # image.convert('mode') 是图像实例对象的一个方法，接受一个 mode 参数，用以指定一种色彩模式
'''
PIL有九种不同模式:
1: 1位像素，黑白，每字节一个像素存储
L: 8位像素，黑白
'''

plt.subplot(221)
plt.imshow(img)
# plt.subplot(222)
# plt.imshow(img_gray, cmap='gray')
plt.subplot(222)
plt.imshow(auto_img_gray, cmap='gray')

# 对灰度图做二值化处理
rows, columns = img_gray.shape
for i in range(rows):
    for j in range(columns):
        if img_gray[i, j] >= 128:
            img_gray[i, j] = 1
        else:
            img_gray[i, j] = 0
print("-----binary_img_gray-----")
print(img_gray)
plt.subplot(223)
plt.imshow(img_gray, cmap='gray')

# 调用接口实现二值化
img_binary = auto_img_gray.convert('1')   # 图像二值化
# pixels = img_binary.load()
# print(pixels)
img_binary_array = np.array(img_binary)
print("-----img_binary_array-----")
print(img_binary_array)
print(img_binary)
print(img_binary.size)
plt.subplot(224)
plt.imshow(img_binary, cmap='gray')     # 显示问题导致手动和调用convert实现二值化的输出图像不同
plt.show()      # 同时显示图像
