import cv2
import numpy as np
import matplotlib.pyplot as plt


a = np.zeros((2, 5))
print('a', a)

img1 = cv2.imread('lena.png')
img2 = plt.imread('lena.png')

plt.subplot(431)
plt.imshow(img1)
plt.title('CV2 Image')

plt.subplot(432)
plt.imshow(img2)
plt.title('matplotlib Image')

imgGray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
plt.subplot(434)
plt.imshow(imgGray1, cmap='gray')
plt.title('Gray')




h, w = img1.shape[:2]
imgR = np.zeros([h, w], img1.dtype)  # 创建一个与img同大小的纯0张量
imgG = np.zeros([h, w], img1.dtype)  # 创建一个与img同大小的纯0张量
imgB = np.zeros([h, w], img1.dtype)  # 创建一个与img同大小的纯0张量
imgGray2 = np.zeros([h, w], img1.dtype)  # 创建一个与img同大小的纯0张量


for i in range(h):
    for j in range(w):
        m = img1[i, j]

        imgR[i, j] = int(m[0] * 0 + m[1] * 0 + m[2])
        imgG[i, j] = int(m[0] * 0 + m[1] + m[2] * 0)
        imgB[i, j] = int(m[0] + m[1] * 0 + m[2] * 0)
        imgGray2[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)



plt.subplot(437)
plt.imshow(imgR, cmap='gray')
plt.title('Red Channel')

plt.subplot(438)
plt.imshow(imgG, cmap='gray')
plt.title('Green Channel')

plt.subplot(439)
plt.imshow(imgB, cmap='gray')
plt.title('Blue Channel')

plt.subplot(435)
plt.imshow(imgGray2, cmap='gray')
plt.title('115903gray Channel')


r, c = imgGray1.shape
for i in range(r):
    for j in range(c):
        if imgGray1[i, j]/255 <= 0.5:
            imgGray1[i, j] = 0
        else:
            imgGray1[i, j] = 1

plt.subplot(4, 3, 10)
plt.imshow(imgGray1, cmap='gray')
plt.title('Binary Image')


plt.tight_layout()
plt.show()
