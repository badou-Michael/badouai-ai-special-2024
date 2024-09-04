from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread("lenna.png")  #opencv 默认 读出来是 BGR 顺序
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)

#for循环实现 灰度化：三通道变单通道，这里采用浮点算法：R 乘系数 0.3 G 乘系数 0.59 B 乘系数 0.11
for i in range(h):
    for j in range(w):
        m = img[i][j]
        img_gray[i][j] = int(0.11*m[0] + 0.59*m[1] + 0.3*m[2])


print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)
cv2.waitKey(0)  #等待一段时间，不要让窗口立刻关闭

#for循环二值化： 非0 即1  有问题！！！ 显示为全黑？？？？？

for i in range(h):
    for j in range(w):
        m = img_gray[i][j] / 255.0
        if(m <= 0.5): img_gray[i][j] = 0
        else :img_gray[i][j] = 1

print("image show binary: %s"%img_gray)
cv2.imshow("image show binary",img_gray)
cv2.waitKey(0)  #等待一段时间，不要让窗口立刻关闭


#plt：自动将读入的图片亮度值归一化，方便后续的二值化
plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("---image lenna----")
print(img)

# 灰度化
img_gray = rgb2gray(img)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

#二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()






