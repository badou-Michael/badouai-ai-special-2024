import cv2
import matplotlib.pyplot as plt
import numpy as np

#灰度化
img = cv2.imread("lenna.png")   #用cv2读取原图
h,w = img.shape[0:2]   #读取图片尺寸
img_gray = np.zeros([h,w], img.dtype)   #创建一个由0组成的图片，[h,w]是shape，img.dtype是数据类型
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

print(m)
print("-------------------------------------")
print(img_gray)
print("-------------------------------------")
cv2.imshow("img_gray", img_gray)

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

plt.subplot(222)
plt.imshow(img_gray, cmap ='gray')

#二值化
img_binary = np.where(img_gray >= 128, 1, 0)
plt.subplot(223)
plt.imshow(img_binary, cmap = 'gray')
print(img_binary)
print("-------------------------------------")

plt.show()
