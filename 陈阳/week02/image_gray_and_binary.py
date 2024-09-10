"""
@author:Ives
@date: 2024/09/05
@description:彩色图片的二值化和灰度化
"""

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 灰度化手写逻辑

image = cv2.imread("lenna.png")
h, w = image.shape[:2] #拿到图片的长和宽
image_gray = np.zeros([h, w], image.dtype)  #赋值给image_gray，这样就是有一张大小一样单通道的图片
for i in range(h):
    for j in range(w):
        m = image[i, j]
        image_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)       #通过计算将image_gray重新赋值实现灰度化

print("m的值为%s"%m)  # 为什么要在外面打印，不能在循环里打印吗？在外面打印不是只能看到最后的值吗？
print(image_gray)
print("image show gray:%s" % image_gray)
cv2.imshow("image gray show", image_gray)

plt.subplot(221)  #在画布上分配一片空间
image = plt.imread("lenna.png")
# img = cv2.imread("lenna.png", False) #也可以使用cv2来读图片
plt.imshow(image)
print("----lenna image----")
print(image)

# 灰度化直接调用接口
image_gray = rgb2gray(image)
# image_gray= cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #如果采用cv2的方式，那就需要将BGR转换为RGB
# image_gray = image #这一段没看懂什么意思，为什么要重新给image_gray赋值image？
plt.subplot(222)
plt.imshow(image_gray, cmap="gray")
print("---image gray---")
print(image_gray)

# 二值化
# 循环方式去二值化
# rows,cols = image_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if image_gray[i,j] <= 0.5:
#             image_gray[i,j] = 0
#         else:
#             image_gray[i,j] = 1

# lambda表达式
image_binary = np.where(image_gray >= 0.5, 1, 0)
print("---image binary---")
print(image_binary)
print(image_binary.shape)

plt.subplot(223)
plt.imshow(image_binary, cmap="gray")
plt.show()
