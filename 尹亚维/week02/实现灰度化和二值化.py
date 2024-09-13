import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

# 灰度化
img = cv2.imread("lenna.png")
h, w = img.shape[:2]  # 获取图片的high和wide
print(f"img high={h}, wide={w}, dtype={img.dtype}")
img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像

print(f"打印最后一次当前high和wide中的BGR坐标={m}")
print(f"得到当前图片的灰度化图片={img_gray}")
cv2.imshow("image show gray", img_gray)

plt.subplot(221)
img = plt.imread("lenna.png")
# img = cv2.imread("lenna.png", False)
plt.imshow(img)
print("---image lenna----")
print(img)

# 灰度化
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)


# 二值化
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i, j] <= 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1

img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----img_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
