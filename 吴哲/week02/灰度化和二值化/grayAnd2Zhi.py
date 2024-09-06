import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

# 灰度化
# gray = R*0.3 + G*0.59 + B*0.11

# cv2读图，读的是BGR
img = cv2.imread("lenna.png")
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        bgr_ij = img[i, j]
        img_gray[i, j] = int(bgr_ij[0] * 0.11 + bgr_ij[1] * 0.59 + bgr_ij[2] * 0.3)

print(img_gray)
cv2.imshow("img show gray", img_gray)
cv2.waitKey(0)

# plt读图，读的是RGB
img = plt.imread("lenna.png")
plt.imshow(img)
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        # 读出来的数值在[0,1]之间
        rgb_ij = img[i, j]
        img_gray[i, j] = int(rgb_ij[0] * 0.3 * 255 + rgb_ij[1] * 0.59 * 255 + rgb_ij[2] * 0.11 * 255)
print(img_gray)
plt.subplot(221)
# 如果不加cmap='gray' 图像展示出来会是绿色的
plt.imshow(img_gray, cmap='gray')

# 直接使用三方灰度化接口，对比上面自己写的接口，体感上三方的图像更柔和，自己写的对比度更高更亮
plt.subplot(222)
img_gray2 = rgb2gray(img)
plt.imshow(img_gray2, cmap='gray')

# 二值化，将灰度化的图进一步以某一阈值划分到两级
img_binary = np.where(img_gray >= 122, 255, 0)
print("------img_binary-------")
print(img_binary)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.waitforbuttonpress()
