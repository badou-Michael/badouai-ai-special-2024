from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


#灰度化
img = cv2.imread("lenna.png")
#获取图片的high和wide
h,w = img.shape[:2]
#创建一张和当前图片大小一样的单通道图片
img_gray = np.zeros([h,w], img.dtype)
for i in range(h):
    for j in range(w):
        # 获取当前high和wide中的BGR坐标
        m = img[i,j]
        # 将BGR坐标转化为gray坐标并复制给新图像
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 +m[2]*0.3)

print(m)
print(img_gray)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("---image lenna---")
print(img)

#灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---iamge gray---")
print(img_gray)


#二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("----image_binary----")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
